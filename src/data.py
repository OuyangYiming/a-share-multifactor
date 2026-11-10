"""BaoStock data and resumable CSV caches for monthly research."""
from pathlib import Path
import socket
import time
import pandas as pd

DATA_START, DATA_END = '2016-12-01', '2025-12-31'
FIELDS = 'date,code,open,high,low,close,preclose,volume,turn,isST,tradestatus'


class _Socket:
    def __init__(self, connection):
        self.connection = connection
        self.deadline = time.monotonic() + 45

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def send(self, payload):
        self.deadline = time.monotonic() + 45
        self.connection.sendall(payload)
        return len(payload)

    def recv(self, count):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('BaoStock response exceeded 45 seconds')
        self.connection.settimeout(min(12, remaining))
        value = self.connection.recv(count)
        if not value:
            raise ConnectionError('BaoStock connection closed')
        return value


class BaoStockClient:
    def __init__(self):
        self.connected = False

    def close(self):
        import baostock.common.context as context
        connection = getattr(context, 'default_socket', None)
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        self.connected = False

    def connect(self):
        import baostock as bs
        import baostock.common.context as context
        import baostock.util.socketutil as socketutil
        self.close()
        socket.setdefaulttimeout(12)
        original_send = socketutil.send_msg
        def guarded_login_send(message):
            if not isinstance(context.default_socket, _Socket):
                context.default_socket = _Socket(context.default_socket)
            return original_send(message)
        socketutil.send_msg = guarded_login_send
        try:
            response = bs.login()
        finally:
            socketutil.send_msg = original_send
        if response.error_code != '0':
            raise ConnectionError(response.error_msg)
        if not isinstance(context.default_socket, _Socket):
            context.default_socket = _Socket(context.default_socket)
        self.connected = True

    def query(self, endpoint, **kwargs):
        import baostock as bs
        for attempt in range(3):
            try:
                if not self.connected:
                    self.connect()
                response = getattr(bs, endpoint)(**kwargs)
                rows = []
                while response.error_code == '0' and response.next():
                    rows.append(response.get_row_data())
                if response.error_code != '0' or len(response.data) == 2000:
                    raise ConnectionError(f'{response.error_code}: {response.error_msg}; incomplete page')
                return pd.DataFrame(rows, columns=response.fields)
            except Exception as exc:
                self.close()
                print(f'Retry {attempt + 1}/3 {endpoint} {kwargs}: {exc}', flush=True)
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        raise RuntimeError('Unreachable')


def _save(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(temp, index=False)
    temp.replace(path)


def _cached(client, path, endpoint, offline=False, **kwargs):
    if path.exists():
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if offline:
        raise FileNotFoundError(f'Missing cache: {path}')
    table = client.query(endpoint, **kwargs)
    _save(table, path)
    return table


def load_calendar(root, client, offline=False):
    pieces = []
    for first, last in [(DATA_START, '2020-12-31'), ('2021-01-01', DATA_END)]:
        path = Path(root) / 'data/mvp/calendar' / f'{first}_{last}.csv'
        pieces.append(_cached(client, path, 'query_trade_dates', offline, start_date=first, end_date=last))
    calendar = pd.concat(pieces, ignore_index=True)
    dates = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_trading_day.eq('1'), 'calendar_date']))
    if dates.has_duplicates or dates.empty:
        raise ValueError('Invalid trading calendar')
    return dates.sort_values()


def load_constituents(root, client, calendar, offline=False):
    days = pd.Series(calendar, index=calendar)
    month_ends = days.groupby(calendar.to_period('M')).max()
    signal_days = month_ends.loc['2017-12':'2025-11']
    frames = []
    for number, date in enumerate(signal_days, 1):
        date_str = date.strftime('%Y-%m-%d')
        for endpoint, expected in [('query_hs300_stocks', 300), ('query_zz500_stocks', 500)]:
            path = Path(root) / 'data/mvp/constituents' / f'{date_str}_{expected}.csv'
            table = _cached(client, path, endpoint, offline, date=date_str)
            if table.empty or table.code.duplicated().any():
                raise ValueError(f'{date_str} {endpoint}: empty or duplicate constituents')
            if len(table) != expected:
                print(f'Constituent count {date_str} {endpoint}: using actual {len(table)}', flush=True)
            if (pd.to_datetime(table.updateDate) > date).any():
                raise ValueError(f'Future snapshot at {date_str}')
            frames.append(pd.DataFrame({'signal_date':date, 'code':table.code, 'index':str(expected)}))
        if number % 6 == 0 or number == len(signal_days):
            print(f'Constituents {number}/{len(signal_days)} months; latest {date_str}', flush=True)
    result = pd.concat(frames, ignore_index=True).drop_duplicates(['signal_date', 'code'])
    _save(result, Path(root) / 'data/mvp/constituents.csv')
    return result


def _price_blocks(start, end, years=4):
    cursor = pd.Timestamp(start)
    end = pd.Timestamp(end)
    while cursor <= end:
        last = min(pd.Timestamp(year=cursor.year + years - 1, month=12, day=31), end)
        yield cursor.strftime('%Y-%m-%d'), last.strftime('%Y-%m-%d')
        cursor = last + pd.Timedelta(days=1)


def load_stock(root, client, code, offline=False):
    folder = Path(root) / 'data/mvp'
    ready = folder / 'daily' / f'{code}.csv'
    if ready.exists():
        return pd.read_csv(ready, parse_dates=['date'], dtype={'code':str})
    pieces = []
    old_blocks = list(_price_blocks(DATA_START, DATA_END, years=4))
    has_old_cache = any((folder / 'raw' / f'{code}_{first}_{last}.csv').exists()
                        for first, last in old_blocks)
    blocks = old_blocks if has_old_cache else _price_blocks(DATA_START, DATA_END, years=5)
    for first, last in blocks:
        path = folder / 'raw' / f'{code}_{first}_{last}.csv'
        pieces.append(_cached(client, path, 'query_history_k_data_plus', offline,
                              code=code, fields=FIELDS, start_date=first, end_date=last,
                              frequency='d', adjustflag='3'))
    table = pd.concat(pieces, ignore_index=True)
    if table.empty:
        raise ValueError(f'No price rows for historical constituent {code}')
    table['date'] = pd.to_datetime(table['date'])
    table = table.sort_values('date').reset_index(drop=True)
    if table.duplicated(['date', 'code']).any():
        raise ValueError(f'Duplicate daily key: {code}')
    for column in FIELDS.split(',')[2:]:
        table[column] = pd.to_numeric(table[column], errors='coerce')
    factors = _cached(client, folder / 'adjustments' / f'{code}.csv', 'query_adjust_factor', offline,
                      code=code, start_date='1990-01-01', end_date=DATA_END)
    if factors.empty:
        table['adj_close'] = table['close']
    else:
        factors['date'] = pd.to_datetime(factors['dividOperateDate'])
        factors['backAdjustFactor'] = pd.to_numeric(factors['backAdjustFactor'], errors='raise')
        factors = factors.sort_values('date').drop_duplicates('date', keep='last')
        aligned = pd.merge_asof(table[['date']], factors[['date', 'backAdjustFactor']], on='date')
        table['adj_close'] = table['close'] * aligned['backAdjustFactor'].fillna(1.0)
    _save(table, ready)
    return table


def load_research_data(root, offline=False, constituents_only=False):
    client = BaoStockClient()
    try:
        calendar = load_calendar(root, client, offline)
        constituents = load_constituents(root, client, calendar, offline)
        if constituents_only:
            return None, constituents, calendar
        codes = sorted(constituents.code.unique())
        print(f'Daily download: {len(codes)} historical constituents, {DATA_START} to {DATA_END}', flush=True)
        frames = []
        started = time.monotonic()
        for number, code in enumerate(codes, 1):
            frame = load_stock(root, client, code, offline)
            frames.append(frame)
            if number % 10 == 0 or number == len(codes):
                elapsed = time.monotonic() - started
                print(f'Daily {number}/{len(codes)}; rows={sum(len(x) for x in frames)}; elapsed={elapsed:.1f}s; last={code}', flush=True)
        return pd.concat(frames, ignore_index=True), constituents, calendar
    finally:
        client.close()
