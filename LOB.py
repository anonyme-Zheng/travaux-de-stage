# %%
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque, defaultdict
from typing import Deque, Dict, List, Optional, Tuple, Any
import bisect
import math
import random
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ============================================================
# 第 1 章：订单与订单簿数据结构
# ============================================================

@dataclass
class Order:
    """基础订单对象（简化版）。

    参数
    ----
    id : 唯一ID（int）
    side : 方向，'B' 买 / 'S' 卖
    price : 价格（以“最小跳动”整数计价更稳；也可用 float，但建议搭配 tick_size 进行换算）
    qty : 剩余可见数量（对 ICEBERG 为“当前显示数量”）
    ts : 到达时间戳（数值越小优先级越高；同价位按 FIFO 排序）
    type : 'LIMIT' | 'MARKET' | 'IOC' | 'ALO' | 'STOP' | 'ICEBERG'
    stop_price : STOP 触发价（可选）
    total_qty : ICEBERG 下单总量（可选，非冰山时可与 qty 相等）
    display_qty : ICEBERG 每次展示量（可选）
    meta : 其他信息（如账户、策略标签等）
    """
    id: int
    side: str  # 'B' or 'S'
    price: Optional[float]
    qty: int
    ts: float
    type: str = "LIMIT"
    stop_price: Optional[float] = None
    total_qty: Optional[int] = None
    display_qty: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def is_buy(self) -> bool:
        return self.side.upper().startswith('B')

    def is_sell(self) -> bool:
        return self.side.upper().startswith('S')

    def is_market(self) -> bool:
        return self.type.upper() == "MARKET"

    def is_ioc(self) -> bool:
        return self.type.upper() == "IOC"

    def is_alo(self) -> bool:
        return self.type.upper() == "ALO"

    def is_stop(self) -> bool:
        return self.type.upper() == "STOP"

    def is_iceberg(self) -> bool:
        return self.type.upper() == "ICEBERG" and self.total_qty is not None and self.display_qty is not None


class PriceLevelQueue:
    """价格层队列（FIFO）。"""
    def __init__(self) -> None:
        self.orders: Deque[Order] = deque()
        self.total_qty: int = 0

    def append(self, order: Order):
        self.orders.append(order)
        self.total_qty += order.qty

    def popleft(self) -> Order:
        o = self.orders.popleft()
        self.total_qty -= o.qty
        return o

    def remove_empty_front(self):
        while self.orders and self.orders[0].qty <= 0:
            self.popleft()

    def __bool__(self):
        return len(self.orders) > 0

    def __len__(self):
        return len(self.orders)


class SideBook:
    """某一侧（买/卖）的订单簿，按价格排序并在价位内维护 FIFO 队列。

    - 买盘：价格从高到低排序
    - 卖盘：价格从低到高排序
    """
    def __init__(self, side: str):
        assert side in ('B','S')
        self.side = side
        self.price_levels: Dict[float, PriceLevelQueue] = {}
        self.prices_sorted: List[float] = []  # 卖盘升序，买盘降序（用负价实现统一 bisect）

    def _key(self, px: float) -> float:
        return -px if self.side == 'B' else px

    def insert(self, order: Order):
        assert not order.is_market(), "市价单不应入簿"
        px = order.price
        if px not in self.price_levels:
            self.price_levels[px] = PriceLevelQueue()
            bisect.insort(self.prices_sorted, self._key(px))
        self.price_levels[px].append(order)

    def best_price(self) -> Optional[float]:
        if not self.prices_sorted:
            return None
        k = self.prices_sorted[0]
        return -k if self.side == 'B' else k

    def pop_best_queue(self) -> Optional[PriceLevelQueue]:
        """弹出最优价位的队列对象（若该价位队列清空，会一并移除）。"""
        best_px = self.best_price()
        if best_px is None:
            return None
        q = self.price_levels[best_px]
        # 不从 dict 中删除，让上层按需处理；但如果空了，要移除价位
        return q

    def cleanup_if_empty(self, px: float):
        q = self.price_levels.get(px)
        if q and len(q) == 0 and q.total_qty == 0:
            # 彻底删除
            del self.price_levels[px]
            # 从排序数组移除
            key = self._key(px)
            idx = bisect.bisect_left(self.prices_sorted, key)
            if 0 <= idx < len(self.prices_sorted) and self.prices_sorted[idx] == key:
                self.prices_sorted.pop(idx)

    def total_depth(self, top_n: int = 1) -> int:
        """返回前 n 档累计数量（可用于计算不平衡度）。"""
        qty = 0
        count = 0
        for k in self.prices_sorted:
            px = -k if self.side == 'B' else k
            q = self.price_levels.get(px)
            if q:
                qty += q.total_qty
                count += 1
                if count >= top_n:
                    break
        return qty


class OrderBook:
    """订单簿（买 + 卖）。支持 FIFO 撮合、特殊订单（IOC/ALO/STOP/ICEBERG）。"""
    def __init__(self, tick_size: float = 0.01):
        self.buy = SideBook('B')
        self.sell = SideBook('S')
        self.tick_size = tick_size
        self._next_id = 1

        # STOP 单在触发前不入簿，单独维护
        self.pending_stop_buys: List[Order] = []
        self.pending_stop_sells: List[Order] = []

        # 追踪最近成交价（用于触发 STOP）；若无成交则用中间价替代
        self.last_trade_price: Optional[float] = None

        # 统计
        self.trades: List[Dict[str, Any]] = []  # 每笔成交记录

    # ---------- 工具与快照 ----------
    def mid(self) -> Optional[float]:
        bp = self.buy.best_price()
        ap = self.sell.best_price()
        if bp is None or ap is None: 
            return None
        return (bp + ap) / 2.0

    def spread(self) -> Optional[float]:
        bp = self.buy.best_price()
        ap = self.sell.best_price()
        if bp is None or ap is None:
            return None
        return ap - bp

    def imbalance(self, top_n: int = 1) -> Optional[float]:
        """I = B_depth / (B_depth + A_depth) in [0,1]"""
        b = self.buy.total_depth(top_n=top_n)
        a = self.sell.total_depth(top_n=top_n)
        if b + a == 0:
            return None
        return b / (b + a)

    # ---------- 下单入口 ----------
    def new_order(self, side: str, price: Optional[float], qty: int, 
                  order_type: str = "LIMIT", ts: Optional[float] = None,
                  stop_price: Optional[float] = None, display_qty: Optional[int] = None, 
                  meta: Optional[Dict[str, Any]] = None) -> Order:
        """统一下单入口。返回订单对象（若 IOC/MARKET 完全成交，order.qty 可能为 0）。"""
        if ts is None:
            ts = time.time()
        if meta is None:
            meta = {}

        oid = self._next_id; self._next_id += 1
        order = Order(
            id=oid, side=side, price=price, qty=qty, ts=ts, type=order_type,
            stop_price=stop_price,
            total_qty=qty if order_type.upper()=="ICEBERG" else None,
            display_qty=display_qty
        )

        if order.is_stop():
            # STOP 不入簿，先挂起等待触发
            if order.is_buy():
                self.pending_stop_buys.append(order)
                self.pending_stop_buys.sort(key=lambda o: o.ts)  # 时间优先
            else:
                self.pending_stop_sells.append(order)
                self.pending_stop_sells.sort(key=lambda o: o.ts)
            return order

        # ALO：价格若会立刻成交，则调整到最优不成交价
        if order.is_alo():
            self._apply_alo_price(order)

        # ICEBERG：用显示量替换 qty，其余进入隐藏余量
        if order.is_iceberg():
            if order.display_qty is None or order.display_qty <= 0:
                raise ValueError("ICEBERG 订单需指定 display_qty>0")
            if order.price is None:
                raise ValueError("ICEBERG 必须是限价单")
            order.qty = min(order.display_qty, order.total_qty)  # 当前可见量

        # 市价或跨价限价：先撮合；否则入簿
        self._match_or_insert(order)

        # 撮合后检查是否触发 STOP（以最近成交价或中间价）
        self._try_trigger_stops()

        return order

    # ---------- 触发规则 ----------
    def _current_trigger_price(self) -> Optional[float]:
        if self.last_trade_price is not None:
            return self.last_trade_price
        return self.mid()

    def _try_trigger_stops(self):
        p = self._current_trigger_price()
        if p is None:
            return

        # 触发 BUY STOP: 价格 >= stop_price
        i = 0
        while i < len(self.pending_stop_buys):
            o = self.pending_stop_buys[i]
            if p >= (o.stop_price or float('inf')):
                # 触发为市价单：常见做法，亦可变成限价（此处用市价）
                trig = Order(id=o.id, side='B', price=None, qty=o.qty, ts=o.ts, type="MARKET")
                self.pending_stop_buys.pop(i)
                self._match_or_insert(trig)
            else:
                i += 1

        # 触发 SELL STOP: 价格 <= stop_price
        i = 0
        while i < len(self.pending_stop_sells):
            o = self.pending_stop_sells[i]
            if p <= (o.stop_price or -float('inf')):
                trig = Order(id=o.id, side='S', price=None, qty=o.qty, ts=o.ts, type="MARKET")
                self.pending_stop_sells.pop(i)
                self._match_or_insert(trig)
            else:
                i += 1

    # ---------- ALO 定价 ----------
    def _apply_alo_price(self, order: Order):
        """若订单价格会与对手方成交，则将其调整为**不成交的最佳价**。"""
        bp = self.buy.best_price()
        ap = self.sell.best_price()
        if order.is_buy():
            # 会与卖盘成交：price >= best ask
            if ap is not None and order.price is not None and order.price >= ap:
                # 调整到 best bid（若无买一，直接取消）
                if bp is None:
                    # 无法提供流动性，取消
                    order.qty = 0
                else:
                    order.price = bp
        else:
            if bp is not None and order.price is not None and order.price <= bp:
                if ap is None:
                    order.qty = 0
                else:
                    order.price = ap

    # ---------- 撮合核心 ----------
    def _match_or_insert(self, order: Order):
        """撮合逻辑（FIFO）：若 order 与对手方可成交则逐价位撮合；否则入簿。"""
        # 市价或跨价限价：match；否则 insert
        if order.is_market() or self._would_cross(order):
            self._execute_against_book(order)
            # 若 IOC 剩余未成交，则取消
            if order.is_ioc() and order.qty > 0:
                order.qty = 0
                return
            # ICEBERG：若仍有隐藏余量，且此次可见量被吃完，刷新一笔新的可见量到队尾
            if order.is_iceberg():
                self._replenish_iceberg(order, after_trade=True)
        else:
            if order.is_market():
                return  # 市价在此不入簿（保护）
            # 将订单放入本方队列
            if order.qty > 0:
                self._insert(order)

    def _insert(self, order: Order):
        side_book = self.buy if order.is_buy() else self.sell
        side_book.insert(order)

    def _would_cross(self, order: Order) -> bool:
        bp = self.buy.best_price()
        ap = self.sell.best_price()
        # 市价或“无价格的 IOC”（视作市价）直接可成交
        if order.is_market() or (order.is_ioc() and order.price is None):
            return True
        if order.price is None:
            return False
        if order.is_buy():
            return ap is not None and order.price >= ap
        else:
            return bp is not None and order.price <= bp

    def _execute_against_book(self, taker: Order):
        """以 taker 身份与对手簿撮合（FIFO）。"""
        contra = self.sell if taker.is_buy() else self.buy
        while taker.qty > 0:
            best_px = contra.best_price()
            if best_px is None:
                break
            # 若是限价且不会再跨价，就停止
            is_market_like = taker.is_market() or (taker.is_ioc() and taker.price is None)
            if (not is_market_like) and (
                (taker.is_buy() and taker.price < best_px) or
                (taker.is_sell() and taker.price > best_px)
            ):
                break

            q = contra.price_levels[best_px]
            # 与最优价位逐笔 FIFO 成交
            while taker.qty > 0 and q:
                top = q.orders[0]
                trade_qty = min(taker.qty, top.qty)
                trade_px = best_px  # 价格以被动方价位成交（常见做法）
                self._record_trade(taker, top, trade_px, trade_qty)
                taker.qty -= trade_qty
                top.qty -= trade_qty
                q.total_qty -= trade_qty
                if top.qty == 0:
                    q.popleft()
                    # 若 top 是 ICEBERG 且有隐藏余量，立刻**自动补足**新的显示量（队尾）
                    if top.is_iceberg() and (top.total_qty - 0) > 0:
                        self._replenish_iceberg(top, after_trade=False)
            # 价位清空，移除
            if len(q) == 0 or q.total_qty == 0:
                # 完整清理该价位
                del contra.price_levels[best_px]
                # 从排序数组移除
                key = contra._key(best_px)
                idx = bisect.bisect_left(contra.prices_sorted, key)
                if 0 <= idx < len(contra.prices_sorted) and contra.prices_sorted[idx] == key:
                    contra.prices_sorted.pop(idx)

    def _replenish_iceberg(self, order: Order, after_trade: bool):
        """补齐 ICEBERG 的下一笔显示量。

        - 对刚被吃完（taker）或作为被动方被吃完（maker）的冰山订单，若仍有隐藏数量，
          则以相同价格和相同订单 ID（实际交易所通常会生成新 ID，此处简化）刷新一笔
          新的可见数量到**本方队列尾部**（丧失时间优先级）。
        """
        if not order.is_iceberg():
            return
        # 已消耗的总量（原 total_qty - 剩余 qty - 已在簿显示的 qty）
        consumed = order.total_qty - order.qty
        remaining_hidden = order.total_qty - consumed
        if remaining_hidden <= 0:
            return
        new_visible = min(order.display_qty or 0, remaining_hidden)
        if new_visible <= 0: 
            return
        # 刷新新的可见段
        new_order = Order(
            id=order.id, side=order.side, price=order.price, qty=new_visible,
            ts=time.time(), type="ICEBERG", stop_price=None,
            total_qty=order.total_qty - consumed,  # 更新剩余总量为“尚未成交”的数量
            display_qty=order.display_qty, meta=order.meta.copy()
        )
        # 放入本方簿尾部
        self._insert(new_order)

    def _record_trade(self, taker: Order, maker: Order, price: float, qty: int):
        self.last_trade_price = price
        self.trades.append({
            "taker_id": taker.id, "maker_id": maker.id,
            "price": price, "qty": qty, "ts": time.time(),
            "side": taker.side  # 以主动方方向记录
        })

    # ---------- 快照导出（用于特征工程） ----------
    def snapshot(self) -> Dict[str, Any]:
        """导出 L1 快照（买一/卖一价与量），便于后续建模。"""
        bp = self.buy.best_price()
        ap = self.sell.best_price()
        bd = self.buy.price_levels[bp].total_qty if bp in self.buy.price_levels else 0
        ad = self.sell.price_levels[ap].total_qty if ap in self.sell.price_levels else 0
        return {"bid1": bp, "ask1": ap, "bid1_qty": bd, "ask1_qty": ad,
                "mid": None if (bp is None or ap is None) else (bp + ap) / 2.0,
                "spread": None if (bp is None or ap is None) else (ap - bp),
                "imbalance": None if (bd + ad) == 0 else bd / (bd + ad)}

# ============================================================
# 第 2 章：价格建模（基于 L1：价差 + 不平衡度）
# ============================================================

def discretize_values(df: pd.DataFrame, tick_size: float, n_spreads: int = 7, n_imbalances: int = 5) -> pd.DataFrame:
    """将价差（以跳动数）与不平衡度离散化为有限桶，以便稳健估计。"""
    out = df.copy()
    out = out.dropna(subset=["bid1","ask1","bid1_qty","ask1_qty"])
    # spread（tick 个数）
    spread_ticks = ((out["ask1"] - out["bid1"]) / tick_size).round().astype(int)
    out["spread_ticks"] = spread_ticks.clip(lower=0)
    # imbalance in [0,1]
    out["imbalance"] = out["bid1_qty"] / (out["bid1_qty"] + out["ask1_qty"]).replace({0: np.nan})
    out["imbalance"] = out["imbalance"].clip(0,1).fillna(0.5)

    # 离散化
    # 价差：按实际取值从小到大截断到最多 n_spreads 桶（超出者归入最后一桶）
    uniq_spreads = sorted(out["spread_ticks"].unique())
    if len(uniq_spreads) > n_spreads:
        # 选取最常见的前 n_spreads-1 桶，其他归最后一桶
        top_vals = out["spread_ticks"].value_counts().index[:n_spreads-1]
        out["spread_bin"] = np.where(out["spread_ticks"].isin(top_vals), out["spread_ticks"], max(uniq_spreads))
    else:
        out["spread_bin"] = out["spread_ticks"]
    # 不平衡：等频分箱
    out["imbalance_bin"] = pd.qcut(out["imbalance"], q=min(n_imbalances, max(1, out["imbalance"].nunique())), labels=False, duplicates="drop")
    out["imbalance_bin"] = out["imbalance_bin"].astype(int)

    return out

def add_forward_return(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """添加未来 h 步的中间价变化（单位：tick），并给出方向标签 {-1,0,1}。"""
    out = df.copy()
    out["mid"] = (out["bid1"] + out["ask1"]) / 2.0
    out["mid_fwd"] = out["mid"].shift(-horizon)
    out["d_mid"] = out["mid_fwd"] - out["mid"]
    out["y"] = np.sign(out["d_mid"]).fillna(0).astype(int)
    return out

def prepare_and_symmetrize(df: pd.DataFrame, horizon: int = 1, tick_size: float = 0.01) -> pd.DataFrame:
    """构造训练数据：
    - 生成未来收益方向标签 y（-1/0/1）
    - 对称化：将 y∈{-1,0,1} 转成二分类 y_up∈{0,1}（涨/不涨）；也可切为三分类。
    """
    out = add_forward_return(df, horizon=horizon).dropna(subset=["spread_bin","imbalance_bin","y"])
    out["y_up"] = (out["y"] > 0).astype(int)
    return out

def fit_price_model(train_df: pd.DataFrame) -> Any:
    """拟合一个简单的逻辑回归模型：预测下一个时刻中间价是否上升 (y_up)。"""
    X = train_df[["spread_bin","imbalance_bin"]].values
    y = train_df["y_up"].values
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    return clf

def predict_up_prob(model, df: pd.DataFrame) -> np.ndarray:
    X = df[["spread_bin","imbalance_bin"]].values
    proba = model.predict_proba(X)[:,1]
    return proba

def microprice(bid, ask, bid_qty, ask_qty) -> float:
    """经典 Microprice 估计：用量加权最优价，作为“即时公允价”近似。"""
    if bid is None or ask is None or bid_qty + ask_qty == 0:
        return np.nan
    w = bid_qty / (bid_qty + ask_qty)
    return w * ask + (1-w) * bid

def price_estimate(df: pd.DataFrame, up_prob: np.ndarray) -> pd.Series:
    """将“上涨概率”与 microprice 结合，得到一个简单的一步价格估计（示意）。"""
    mp = df.apply(lambda r: microprice(r["bid1"], r["ask1"], r["bid1_qty"], r["ask1_qty"]), axis=1)
    # 用上涨概率对最优价做凸组合（仅作演示；可替换成更严谨的离散状态价格模型）
    est = up_prob * df["ask1"].values + (1 - up_prob) * df["bid1"].values
    # 再与 microprice 取平均，降低噪声
    est = 0.5 * est + 0.5 * mp.values
    return pd.Series(est, index=df.index)

# ============================================================
# 第 3 章：示例 - 随机流模拟 + 模型训练与评估
# ============================================================

def simulate_random_flow(n_steps: int = 2000, seed: int = 7, tick_size: float = 0.01) -> pd.DataFrame:
    """生成一个简易 L1 序列：通过 LOB 撮合引擎注入随机订单流。"""
    rng = random.Random(seed)
    ob = OrderBook(tick_size=tick_size)
    # 初始挂单：构造一个对称的书
    mid0 = 10.00
    for i in range(5):
        ob.new_order('B', price=mid0 - (i+1)*tick_size, qty=100*(5-i), order_type="LIMIT", ts=i)
        ob.new_order('S', price=mid0 + (i+1)*tick_size, qty=100*(5-i), order_type="LIMIT", ts=i)

    rows = []
    ts = 0.0
    for t in range(n_steps):
        ts += 0.01
        # 决定事件类型：限价、市场、冰山、IOC、ALO、STOP
        u = rng.random()
        if u < 0.65:
            # 新限价单
            side = 'B' if rng.random()<0.5 else 'S'
            best = ob.buy.best_price() if side=='B' else ob.sell.best_price()
            if best is None:
                price = ob.mid() or mid0
            else:
                # 在最优价附近 ±2 tick
                delta = rng.choice([-2,-1,0,1,2])*tick_size
                price = max(tick_size, best + (delta if side=='B' else -delta))
            qty = rng.choice([50,100,150,200])
            typ = rng.choices(["LIMIT","ICEBERG"], weights=[0.8,0.2])[0]
            if typ=="ICEBERG":
                ob.new_order(side, price, qty, order_type="ICEBERG", display_qty=max(10, qty//5), ts=ts)
            else:
                ob.new_order(side, price, qty, order_type="LIMIT", ts=ts)
        elif u < 0.85:
            # 市价/IOC
            side = 'B' if rng.random()<0.5 else 'S'
            qty = rng.choice([50,80,120,200])
            typ = rng.choices(["MARKET","IOC"], weights=[0.6,0.4])[0]
            ob.new_order(side, price=None, qty=qty, order_type=typ, ts=ts)
        elif u < 0.95:
            # ALO：若会成交则自动改价
            side = 'B' if rng.random()<0.5 else 'S'
            best = ob.buy.best_price() if side=='B' else ob.sell.best_price()
            if best is None:
                continue
            # 刻意给出会跨价的价格，测试 ALO 逻辑
            price = (ob.sell.best_price() if side=='B' else ob.buy.best_price())
            qty = rng.choice([50,100,150])
            ob.new_order(side, price=price, qty=qty, order_type="ALO", ts=ts)
        else:
            # STOP：触发后转市价单
            side = 'B' if rng.random()<0.5 else 'S'
            best = ob.sell.best_price() if side=='B' else ob.buy.best_price()
            if best is None: continue
            # 设置略微偏离当前最优的触发价
            stop_px = best + (2*tick_size if side=='B' else -2*tick_size)
            qty = rng.choice([50,100])
            ob.new_order(side, price=None, qty=qty, order_type="STOP", stop_price=stop_px, ts=ts)

        snap = ob.snapshot()
        snap["t"] = t
        rows.append(snap)

    df = pd.DataFrame(rows).ffill().dropna()
    return df

def train_and_evaluate_on_df(df: pd.DataFrame, tick_size: float = 0.01, horizon: int = 1, 
                             train_ratio: float = 0.8, n_spreads: int = 7, n_imbalances: int = 5):
    df_disc = discretize_values(df, tick_size=tick_size, n_spreads=n_spreads, n_imbalances=n_imbalances)
    df_prep = prepare_and_symmetrize(df_disc, horizon=horizon, tick_size=tick_size)
    # 划分训练/测试
    split = int(len(df_prep) * train_ratio)
    train_df = df_prep.iloc[:split].copy()
    test_df = df_prep.iloc[split:].copy()

    model = fit_price_model(train_df)
    proba = predict_up_prob(model, test_df)
    est = price_estimate(test_df, proba)

    # 报告分类精度（涨/不涨）
    y_true = test_df["y_up"].values
    y_pred = (proba >= 0.5).astype(int)
    report = classification_report(y_true, y_pred, output_dict=True)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    out = {
        "model": model,
        "test_df": test_df,
        "proba_up": proba,
        "price_est": est,
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
    }
    return out

# ============================================================
# 第 4 章：简单报价策略（演示）
# ============================================================

def demo_quote_logic(row: pd.Series, est_px: float, tick_size: float = 0.01) -> Tuple[float, float]:
    """基于预测价格决定挂单：
    - 若 est > mid 则做多倾向：挂买在 bid1，卖在 ask1+1tick（被动卖）
    - 若 est < mid 则做空倾向：挂卖在 ask1，买在 bid1-1tick
    仅作为演示，实际可加入库存与风险约束。
    """
    bid, ask = row["bid1"], row["ask1"]
    mid = (bid + ask)/2.0
    if est_px >= mid:
        buy_px = bid
        sell_px = ask + tick_size
    else:
        buy_px = bid - tick_size
        sell_px = ask
    return buy_px, sell_px

# ============================================================
# 入口：演示运行
# ============================================================

def main_demo():
    print(">>> 模拟随机订单流并训练价格模型 ...")
    tick = 0.01
    df = simulate_random_flow(n_steps=3000, seed=42, tick_size=tick)
    res = train_and_evaluate_on_df(df, tick_size=tick, horizon=1, train_ratio=0.8,
                                   n_spreads=7, n_imbalances=5)
    print(f"测试集 上涨/不涨 准确率：{res['accuracy']:.3f}")
    # 给出若干行示例的报价建议
    print(">>> 示例报价（前 5 条测试样本）：")
    test_df = res["test_df"].copy()
    test_df = test_df.assign(est=res["price_est"].values, proba_up=res["proba_up"])
    for i, row in test_df.head(5).iterrows():
        buy_px, sell_px = demo_quote_logic(row, row["est"], tick_size=tick)
        print(f"t={int(row['t'])}  bid1={row['bid1']:.2f} ask1={row['ask1']:.2f}  "
              f"p_up={row['proba_up']:.2f}  est={row['est']:.2f}  -> 建议挂单: 买 {buy_px:.2f} / 卖 {sell_px:.2f}")

if __name__ == "__main__":
    main_demo()


# %%


# %%


# %%
import sys
!{sys.executable} -m pip install --upgrade pip
!{sys.executable} -m pip install "plotly>=5"

# %%
import plotly.io as pio
# Jupyter(含JupyterLab)： 
pio.renderers.default = "notebook_connected"
# 如果你在 Colab： pio.renderers.default = "colab"
# VS Code： pio.renderers.default = "vscode"
# 纯脚本/终端想用浏览器弹窗： pio.renderers.default = "browser"

# %%
import plotly.express as px
import pandas as pd
import numpy as np

# Range of prices in plot
price_range = [x for x in range(95,105)]

# Buy limit orders
buy_LOs = [0,0,0,1,1,0,0,0,0,0]

# Sell limit orders
sell_LOs = [0,0,0,0,0,0,2,0,0,0]

# Forming a dataframe, and convertin it into long form for plotting
LOB_dict = {"prices":price_range, "buy":buy_LOs, "sell":sell_LOs}
LOB_df = pd.DataFrame(LOB_dict)
longer_LOB = pd.melt(LOB_df, id_vars = ["prices"], value_vars = ["buy","sell"], var_name = "LO_type", value_name = "n_orders")
print(longer_LOB.head(3)) # first 3 prices
print(longer_LOB.tail(3)) # first 3 prices
fig = px.bar(longer_LOB, x = "prices", y = "n_orders", color = "LO_type")
fig.show()

# %%
import plotly.express as px
import pandas as pd
import numpy as np

# 价格档位（95 到 104）
price_range = [x for x in range(95, 105)]

# ---- 练习 2.1：补全这里 ----
# 初始买单：98、99 各 1；加入订单 5（100 的买单）→ 100 档 +1
buy_LOs  = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]

# 初始卖单：101 有 2；加入订单 6（102 的卖单）→ 102 档 +1
sell_LOs = [0, 0, 0, 0, 0, 0, 2, 1, 0, 0]
# --------------------------------

def plot_LOB(price_range, buy_LOs, sell_LOs):
    n_prices = len(price_range)

    # 输入合法性检查
    if n_prices != len(buy_LOs):
        return (f"Error: Different numbers of prices and LOs. "
                f"Received {n_prices} prices and {len(buy_LOs)} buy limit orders")
    elif n_prices != len(sell_LOs):
        return (f"Error: Different numbers of prices and LOs. "
                f"Received {n_prices} prices and {len(sell_LOs)} buy limit orders")

    # 组装数据框并转成长表（long form）以便绘图
    lob_df = pd.DataFrame({
        "prices": price_range,
        "buy":   buy_LOs,
        "sell":  sell_LOs
    })
    longer = pd.melt(
        lob_df,
        id_vars=["prices"],
        value_vars=["buy", "sell"],
        var_name="LO_type",
        value_name="n_orders"
    )

    # 绘制柱状图（Plotly）
    fig = px.bar(longer, x="prices", y="n_orders", color="LO_type",
                 title="Limit Order Book")
    return fig

# 运行查看
plot_LOB(price_range, buy_LOs, sell_LOs).show()


# %%
# We import the unittest module for testing
import unittest
# The datetime module is used to store order times
import datetime as dt
# The bisect module is used to perform some of the sorting tasks needed.
import bisect
from typing import Dict, List, Any, Optional, Tuple

# ---------------------------
# Implementations (fill-ins)
# ---------------------------

LOB = Dict[str, List[Any]]

def _n_rows(side_dict: LOB) -> int:
    if not side_dict:
        return 0
    # assume all columns same length
    return len(next(iter(side_dict.values())))

def insertRowInPosition(side_dict: LOB, new_row: Dict[str, Any], index: int) -> LOB:
    """
    Insert new_row at 'index' across all columns in side_dict.
    - Valid index range: 0..n (inclusive). index==n 表示追加到末尾。
    - new_row 必须至少包含 side_dict 里已有的所有列（通常为 ID/Price/Time）。
    - 就地修改并返回 side_dict。
    """
    n = _n_rows(side_dict)
    if index < 0 or index > n:
        raise ValueError(f"Invalid index {index}; allowed 0..{n}")

    # 初始化空字典的情况
    if n == 0:
        required = side_dict.keys() if side_dict else ["ID", "Price", "Time"]
        for k in required:
            if k not in new_row:
                raise TypeError(f"new_row is missing field '{k}'")
        # 创建列
        for k in required:
            side_dict[k] = [new_row[k]]
        return side_dict

    # 非空：检查字段齐全
    for k in side_dict.keys():
        if k not in new_row:
            raise TypeError(f"new_row is missing field '{k}'")

    # 逐列插入
    for k in side_dict.keys():
        side_dict[k].insert(index, new_row[k])
    return side_dict

def getSortIdx(
    side_dict: LOB,
    new_row: Dict[str, Any],
    sort_1: str,
    sort_2: Optional[str] = None,
    reverse_1: bool = False,
    reverse_2: bool = False
) -> int:
    """
    返回把 new_row 插入到 side_dict 中以保持排序所需的位置。
    - sort_1 为主键；sort_2（可选）为次键；
    - reverse_* = True 表示该键按降序排列。
    - 稳定插入：当主键（及次键）都相等时，插在现有相等元素之后（bisect_right 行为）。
    """
    n = _n_rows(side_dict)
    if n == 0:
        return 0

    v1 = new_row[sort_1]
    v2 = new_row[sort_2] if sort_2 is not None else None

    for i in range(n):
        a1 = side_dict[sort_1][i]
        if v1 != a1:
            # 升序：v1 < a1 时应插在 i 前；降序：v1 > a1 时应插在 i 前
            if (not reverse_1 and v1 < a1) or (reverse_1 and v1 > a1):
                return i
            else:
                continue
        # 主键相等，用次键
        if sort_2 is not None:
            a2 = side_dict[sort_2][i]
            if v2 != a2:
                if (not reverse_2 and v2 < a2) or (reverse_2 and v2 > a2):
                    return i
                else:
                    continue
        # 主键与次键都相等 → 放到后面
    return n

def deleteRow(side_dict: LOB, index: int) -> LOB:
    """
    删除指定行；index 必须在 [0, n-1]。
    就地修改并返回 side_dict。
    """
    n = _n_rows(side_dict)
    if index < 0 or index >= n:
        raise ValueError(f"Invalid index {index}; allowed 0..{n-1}")
    for k in list(side_dict.keys()):
        side_dict[k].pop(index)
    return side_dict

def addToDict(side_dict: LOB, new_row: Dict[str, Any], side: str) -> LOB:
    """
    将新订单按既定排序插入：
      - Buy：Price 降序，其次 Time 升序
      - Sell：Price 升序，其次 Time 升序
    就地修改并返回 side_dict。
    """
    if side not in {"Buy", "Sell"}:
        raise ValueError("side must be 'Buy' or 'Sell'")
    reverse_1 = (side == "Buy")  # 买单价格高优先 → 降序
    idx = getSortIdx(side_dict, new_row, "Price", "Time", reverse_1=reverse_1, reverse_2=False)
    return insertRowInPosition(side_dict, new_row, idx)

def swapDictSides(dict_1: LOB, dict_2: LOB, order_side: str) -> Tuple[LOB, LOB]:
    """根据新订单方向返回 (同侧字典, 对手盘字典)。"""
    if order_side == "Buy":
        return dict_1, dict_2
    if order_side == "Sell":
        return dict_2, dict_1
    raise ValueError("order_side must be 'Buy' or 'Sell'")

def checkCrossesSpread(order_price: float, best_price: Optional[float], order_side: str) -> bool:
    """
    判断是否“跨越价差”：
      - Buy：order_price >= best_ask → True
      - Sell：order_price <= best_bid → True
      - 若 best_price 为 None（对手盘为空）→ False
    """
    if best_price is None:
        return False
    if order_side == "Buy":
        return order_price >= best_price
    if order_side == "Sell":
        return order_price <= best_price
    raise ValueError("order_side must be 'Buy' or 'Sell'")

# ---------------------------
# Unit tests (with the missing ones added)
# ---------------------------

class TestHelperFunctions(unittest.TestCase):
    def test_insertRowInPosition_zeroIndexvalid_newDict(self):
        input_dict = {"ID":[2,3], "Price":[99,98], "Time":[dt.datetime(2024,9,1,10,30,1), dt.datetime(2024,9,1,10,30,4)]}
        new_row = {"Type":"Buy", "ID":5, "Price":100, "Time":dt.datetime(2024,9,1,10,30,11)}
        index = 0
        expected_dict = {"ID":[5,2,3], "Price":[100,99,98], "Time":[dt.datetime(2024,9,1,10,30,11),dt.datetime(2024,9,1,10,30,1), dt.datetime(2024,9,1,10,30,4)]}
        output_dict = insertRowInPosition(input_dict, new_row, index)
        self.assertDictEqual(output_dict,expected_dict)
    
    def test_insertRowInPosition_validIndex_newDict(self):
        input_dict = {"ID":[1,4], "Price":[100,100], "Time":[dt.datetime(2024,9,1,10,30,3), dt.datetime(2024,9,1,10,30,10)]}
        new_row = {"Type":"Sell", "ID":6, "Price":102, "Time":dt.datetime(2024,9,1,10,30,12)}
        index = 2
        expected_dict = {"ID":[1,4,6], "Price":[100,100,102], "Time":[dt.datetime(2024,9,1,10,30,3),dt.datetime(2024,9,1,10,30,10), dt.datetime(2024,9,1,10,30,12)]}
        output_dict = insertRowInPosition(input_dict, new_row, index)
        self.assertDictEqual(output_dict,expected_dict)
    
    def test_insertRowInPosition_invalidIndex_ValueError(self):
        input_dict = {"ID":[2,3], "Price":[99,98], "Time":[dt.datetime(2024,9,1,10,30,1), dt.datetime(2024,9,1,10,30,4)]}
        new_row = {"Type":"Buy", "ID":5, "Price":100, "Time":dt.datetime(2024,9,1,10,30,11)}
        index = 3
        expected_exception = ValueError
        self.assertRaises(expected_exception, insertRowInPosition, input_dict, new_row, index)

    # ---- Missing field test (新增) ----
    def test_insertRowInPosition_missingField_TypeError(self):
        input_dict = {"ID":[2], "Price":[99], "Time":[dt.datetime(2024,9,1,10,30,1)]}
        # 缺少 Time 字段
        new_row = {"Type":"Buy", "ID":5, "Price":100}
        with self.assertRaises(TypeError):
            insertRowInPosition(input_dict, new_row, 1)
    # ----------------------------------

    def test_getSortIdx_singleSort_sortedIndex(self):
        input_dict = {"ID":[2,3], "Price":[99,98], "Time":[dt.datetime(2024,9,1,10,30,1), dt.datetime(2024,9,1,10,30,4)]}
        new_row = {"Type":"Buy","ID":5, "Price":99, "Time":dt.datetime(2024,9,1,10,30,3)}
        sort_1 = "Time"
        expected_index = 1 # 3 应插在 1 和 4 之间
        output_index = getSortIdx(input_dict, new_row, sort_1)
        self.assertEqual(expected_index, output_index)

    def test_getSortIdx_doubleSort_sortedIndex(self):
        input_dict = {"ID":[1,4,5], "Price":[100,100,101], "Time":[dt.datetime(2024,9,1,10,30,3), dt.datetime(2024,9,1,10,30,10),dt.datetime(2024,9,1,10,30,6)]}
        new_row = {"Type":"Sell", "ID":6, "Price":100, "Time":dt.datetime(2024,9,1,10,30,7)}
        sort_1 = "Price"; sort_2 = "Time"
        expected_index = 1
        output_index = getSortIdx(input_dict, new_row, sort_1, sort_2)
        self.assertEqual(expected_index, output_index)
    
    def test_getSortIdx_doubleReversedSort_sortedIndex(self):
        input_dict = {"ID":[2,3], "Price":[99,98], "Time":[dt.datetime(2024,9,1,10,30,1), dt.datetime(2024,9,1,10,30,4)]}
        new_row = {"Type":"Buy","ID":5, "Price":97, "Time":dt.datetime(2024,9,1,10,30,3)}
        sort_1 = "Price"; reverse_1 = True
        expected_index = 2
        output_index = getSortIdx(input_dict, new_row, sort_1, reverse_1 = reverse_1)
        self.assertEqual(expected_index, output_index)
    
    def test_deleteRow_validIndex_newDict(self):
        input_dict = {"ID":[2,3], "Price":[99,98], "Time":[dt.datetime(2024,9,1,10,30,1), dt.datetime(2024,9,1,10,30,4)]}
        delete_index = 1
        expected_dict = {"ID":[2], "Price":[99], "Time":[dt.datetime(2024,9,1,10,30,1)]}
        output_dict = deleteRow(input_dict, delete_index)
        self.assertDictEqual(output_dict, expected_dict)
    
    def test_deleteRow_invalidIndex_ValueError(self):
        input_dict = {"ID":[2,3], "Price":[99,98], "Time":[dt.datetime(2024,9,1,10,30,1), dt.datetime(2024,9,1,10,30,4)]}
        delete_index = 2
        expected_exception = ValueError
        self.assertRaises(expected_exception, deleteRow, input_dict, delete_index)

    def test_addToDict_buyOrder_newDict(self):
        input_dict = {"ID":[2,3], "Price":[99,98], "Time":[dt.datetime(2024,9,1,10,30,1), dt.datetime(2024,9,1,10,30,4)]}
        new_order = {"Type": "Buy", "ID":5,"Price": 100, "Time":dt.datetime(2024,9,1,10,30,10) }
        order_side = "Buy"
        expected_dict = {"ID":[5,2,3], "Price":[100,99,98], 
                         "Time":[dt.datetime(2024,9,1,10,30,10),dt.datetime(2024,9,1,10,30,1), dt.datetime(2024,9,1,10,30,4)]}
        addToDict(input_dict, new_order, order_side)
        self.assertDictEqual(input_dict, expected_dict)

    def test_addToDict_sellOrder_newDict(self):
        input_dict = {"ID":[1,4,5], "Price":[100,100,101], 
                      "Time":[dt.datetime(2024,9,1,10,30,3), dt.datetime(2024,9,1,10,30,10),dt.datetime(2024,9,1,10,30,6)]}
        new_order = {"Type":"Sell", "ID":6, "Price":101, "Time":dt.datetime(2024,9,1,10,30,4)}
        order_side = "Sell"
        expected_dict = {"ID":[1,4,6,5], "Price":[100,100,101,101], 
                        "Time":[dt.datetime(2024,9,1,10,30,3), dt.datetime(2024,9,1,10,30,10),
                                dt.datetime(2024,9,1,10,30,4),dt.datetime(2024,9,1,10,30,6)]}
        addToDict(input_dict, new_order, order_side)
        self.assertDictEqual(input_dict, expected_dict)
    
    # ---- Multiple-order test (新增) ----
    def test_addToDict_multipleOrders_backToBack(self):
        input_dict = {"ID":[2,3], "Price":[99,98],
                      "Time":[dt.datetime(2024,9,1,10,30,1), dt.datetime(2024,9,1,10,30,4)]}
        o1 = {"Type":"Buy","ID":5,"Price":100,"Time":dt.datetime(2024,9,1,10,30,10)}
        o2 = {"Type":"Buy","ID":6,"Price":97,"Time":dt.datetime(2024,9,1,10,30,12)}
        addToDict(input_dict, o1, "Buy")
        addToDict(input_dict, o2, "Buy")
        expected = {"ID":[5,2,3,6], "Price":[100,99,98,97],
                    "Time":[dt.datetime(2024,9,1,10,30,10),
                            dt.datetime(2024,9,1,10,30,1),
                            dt.datetime(2024,9,1,10,30,4),
                            dt.datetime(2024,9,1,10,30,12)]}
        self.assertDictEqual(input_dict, expected)
    # -----------------------------------

    def test_swapDictSides_buyOrder_twoDicts(self):
        buy_dict = {"ID":[2,3], "Price":[99,98], "Time":[dt.datetime(2024,9,1,10,30,1), dt.datetime(2024,9,1,10,30,4)]}
        sell_dict = {"ID":[1,4], "Price":[100,100], "Time":[dt.datetime(2024,9,1,10,30,3), dt.datetime(2024,9,1,10,30,10)]}
        order_side = "Buy"
        same_dict, opposing_dict = swapDictSides(buy_dict, sell_dict, order_side)
        self.assertDictEqual(buy_dict, same_dict)
        self.assertDictEqual(sell_dict, opposing_dict)

    # ---- Swap for Sell side (新增) ----
    def test_swapDictSides_sellOrder_twoDicts(self):
        buy_dict = {"ID":[2,3], "Price":[99,98], "Time":[dt.datetime(2024,9,1,10,30,1), dt.datetime(2024,9,1,10,30,4)]}
        sell_dict = {"ID":[1,4], "Price":[100,100], "Time":[dt.datetime(2024,9,1,10,30,3), dt.datetime(2024,9,1,10,30,10)]}
        order_side = "Sell"
        same_dict, opposing_dict = swapDictSides(buy_dict, sell_dict, order_side)
        self.assertDictEqual(sell_dict, same_dict)
        self.assertDictEqual(buy_dict, opposing_dict)
    # ----------------------------------

    def test_checkCrossesSpread_buyOrder_True(self):
        order_price = 10
        best_price_val = 9 # Lowest sell price
        order_side = "Buy"
        expected_output = True
        crosses_spread = checkCrossesSpread(order_price, best_price_val, order_side)
        self.assertEqual(expected_output, crosses_spread)
    
    def test_checkCrossesSpread_buyOrder_False(self):
        order_price = 8
        best_price_val = 9
        order_side = "Buy"
        expected_output = False
        crosses_spread = checkCrossesSpread(order_price, best_price_val, order_side)
        self.assertEqual(expected_output, crosses_spread)

    # ---- Sell side tests (新增) ----
    def test_checkCrossesSpread_sellOrder_True(self):
        order_price = 9      # 卖价 <= 最优买价 → 可成交
        best_bid = 10
        self.assertTrue(checkCrossesSpread(order_price, best_bid, "Sell"))

    def test_checkCrossesSpread_sellOrder_False(self):
        order_price = 11     # 卖价 > 最优买价 → 不成交
        best_bid = 10
        self.assertFalse(checkCrossesSpread(order_price, best_bid, "Sell"))
    # --------------------------------

    # ---- Equality case (新增) ----
    def test_checkCrossesSpread_equalPrice_buy_True(self):
        # 买单价格等于最优卖价 → 视为跨越（可成交）
        self.assertTrue(checkCrossesSpread(9, 9, "Buy"))
    # 也可以加一个 sell 等价测试（可选）
    def test_checkCrossesSpread_equalPrice_sell_True(self):
        self.assertTrue(checkCrossesSpread(10, 10, "Sell"))
    # --------------------------------


# %%
def helperFunctionSuite():

    loader = unittest.TestLoader()
    helper_funtions_suite = loader.loadTestsFromTestCase(TestHelperFunctions)
    return helper_funtions_suite

runner = unittest.TextTestRunner()

# Running the tests, we see every test fails.
runner.run(helperFunctionSuite())

# %%
def insertRowInPosition(side_dict, new_row, index):
    # Check if the index is valid:
    n_orders = len(side_dict["ID"])
    if index >= n_orders + 1 or index < 0:
        raise ValueError("Index not in range")
    
    # Check if the order has the required fields
    fields_in_dict = set(side_dict.keys())
    # -------- 补全开始
    # 新行必须至少包含字典中已有的全部列（可多出 "Type" 等字段，但不能缺）
    missing = fields_in_dict - set(new_row.keys())
    if missing:
        raise TypeError(f"new_row is missing field(s): {missing}")

    # 逐列在同一 index 位置插入（就地修改）
    for k in side_dict.keys():
        side_dict[k].insert(index, new_row[k])
    # -------- 补全结束
    return side_dict


def getSortIdx(side_dict, new_row, sort_1, sort_2 = None, reverse_1 = False, reverse_2 = False):
    # get the position to insert the new row based on the (up to 2) sort column names:
    sort_1_list = side_dict[sort_1]
    n_rows = len(sort_1_list)
    # Save the first and last position that the new row could be inserted into.
    if reverse_1:
        sort_1_list.reverse()
        rev_lower_index = bisect.bisect_left(sort_1_list, new_row[sort_1])
        rev_upper_index = bisect.bisect_right(sort_1_list, new_row[sort_1])
        sort_1_list.reverse()
        upper_index = n_rows - rev_lower_index
        lower_index = n_rows - rev_upper_index
    else:
        lower_index = bisect.bisect_left(sort_1_list, new_row[sort_1])
        upper_index = bisect.bisect_right(sort_1_list, new_row[sort_1])

    # -------- 补全开始
    # 在主键相等的区间 [lower_index, upper_index) 内，再按次键决定相对位置
    if sort_2 is None or lower_index == upper_index:
        sub_index = 0
    else:
        second_slice = side_dict[sort_2][lower_index:upper_index]
        v2 = new_row[sort_2]
        # 若 reverse_2=False：升序 → 放在 <= 的元素之后（等价 bisect_right）
        # 若 reverse_2=True ：降序 → 放在 >= 的元素之后（保持稳定）
        sub_index = 0
        if not reverse_2:
            for val in second_slice:
                if v2 >= val:
                    sub_index += 1
                else:
                    break
        else:
            for val in second_slice:
                if v2 <= val:
                    sub_index += 1
                else:
                    break
    # -------- 补全结束
    return(lower_index + sub_index)


def deleteRow(side_dict, index):
    n_orders = len(side_dict["ID"])
    # ------ 补全开始
    if index < 0 or index >= n_orders:
        raise ValueError("Index not in range")
    for k in list(side_dict.keys()):
        side_dict[k].pop(index)
    # ------ 补全结束
    return side_dict


def addToDict(side_dict, new_order, order_side):
    if order_side == "Sell":
        new_idx = getSortIdx(side_dict, new_order,"Price","Time")
    # ------ 补全开始
    elif order_side == "Buy":
        # 买单按价格降序、时间升序
        new_idx = getSortIdx(side_dict, new_order, "Price", "Time", reverse_1=True)
    else:
        raise ValueError("order_side must be 'Buy' or 'Sell'")

    insertRowInPosition(side_dict, new_order, new_idx)
    # ------ 补全结束
    return side_dict


def swapDictSides(buy_dict, sell_dict, order_side):
    # ------ 补全开始
    if order_side == "Buy":
        return buy_dict, sell_dict   # (同侧, 对手盘)
    elif order_side == "Sell":
        return sell_dict, buy_dict
    else:
        raise ValueError("order_side must be 'Buy' or 'Sell'")
    # ------ 补全结束


def checkCrossesSpread(order_price, best_price, order_side):
    # ------ 补全开始
    # 若对手盘为空，则无法成交
    if best_price is None:
        return False
    if order_side == "Buy":
        return order_price >= best_price   # 买价 ≥ 卖一 → 成交
    elif order_side == "Sell":
        return order_price <= best_price   # 卖价 ≤ 买一 → 成交
    else:
        raise ValueError("order_side must be 'Buy' or 'Sell'")
    # ------- 补全结束


# %%
runner.run(helperFunctionSuite())

# %% [markdown]
# #2.4

# %%
def checkIfMatch(same_side_dict, opposing_dict, new_order):
    # 取出对手盘（与新单相反方向）的价格列表
    opposing_prices = opposing_dict["Price"]
    # 新单的价格与方向（方向用来判断是否“跨越价差”）
    order_price = new_order["Price"]
    order_side  = new_order["Side"]

    # ——【关键修正】——
    # 若对手盘为空（例如第一张订单进来时），不可能成交：
    #   直接将新单加入同侧订单簿，并返回 do_matching = False
    if len(opposing_prices) == 0:
        addToDict(same_side_dict, new_order, order_side)
        return (same_side_dict, opposing_dict, None, False)

    # 对手盘非空：最优价按“索引 0”约定直接取即可
    best_price_index = 0
    best_price = opposing_prices[best_price_index]

    # 判断是否“跨越价差”：Buy: 价格 ≥ 卖一；Sell: 价格 ≤ 买一
    need_to_match = checkCrossesSpread(order_price, best_price, order_side)

    if not need_to_match:
        # 不需要撮合 → 新单加入同侧订单簿
        addToDict(same_side_dict, new_order, order_side)
        return (same_side_dict, opposing_dict, None, False)
    else:
        # 需要撮合 → 暂不改动字典（由外层真正执行删除等动作）
        return (same_side_dict, opposing_dict, new_order, True)


def priceTimeMatching1(buy_dict, sell_dict, new_order):
    # 新单方向
    order_side = new_order["Side"]

    # 将（买簿, 卖簿）重排成（同侧簿, 对手簿），便于统一处理
    same_side_dict, opposing_dict = swapDictSides(buy_dict, sell_dict, order_side)

    # 先判断是否需要撮合；若不需要，函数已经把新单加入同侧簿
    same_side_dict, opposing_dict, new_order, do_matching = checkIfMatch(
        same_side_dict, opposing_dict, new_order
    )

    if not do_matching:
        # 不撮合：把顺序换回（返回值始终是 buy_dict, sell_dict）
        return swapDictSides(same_side_dict, opposing_dict, order_side)
    else:
        # 需要撮合：删除**对手盘**的最优一单（索引为 0）
        # （修复：原代码误删了 same_side_dict 的第 0 行）
        deleteRow(opposing_dict, 0)
        # 返回时同样把顺序换回
        return swapDictSides(same_side_dict, opposing_dict, order_side)


# %%
class TestPriceTime1(unittest.TestCase):

    def test_priceTimeMatching1_noOpposingBuyOrder_addsToDict(self):
        # 买簿里已有两张买单，卖簿为空
        buy_dict  = {"ID":[2,3], "Price":[99,98],
                     "Time":[dt.datetime(2024,9,1,10,30,1),
                             dt.datetime(2024,9,1,10,30,4)]}
        sell_dict = {"ID":[], "Price":[], "Time":[]}

        # 新来一张买单（对手盘不存在 → 应直接加入买簿）
        new_order = {"Side":"Buy", "ID":5, "Price":100,
                     "Time":dt.datetime(2024,9,1,10,30,10)}

        # 期望：新单按“买侧：价降序/时升序”排在最前
        expected_buy_dict  = {"ID":[5,2,3], "Price":[100,99,98],
                              "Time":[dt.datetime(2024,9,1,10,30,10),
                                      dt.datetime(2024,9,1,10,30,1),
                                      dt.datetime(2024,9,1,10,30,4)]}
        expected_sell_dict = {"ID":[], "Price":[], "Time":[]}

        # 调用原型算法
        new_buy_dict, new_sell_dict = priceTimeMatching1(buy_dict, sell_dict, new_order)

        # 断言：卖簿不变；买簿多了新单且顺序正确
        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict,  expected_buy_dict)

    # —— 额外测试 1：空空订单簿上的第一张单（对应提示里的情形） ——
    def test_priceTimeMatching1_firstOrderOnEmptyLOB_addsToCorrectSide(self):
        # 双边都为空
        buy_dict  = {"ID":[], "Price":[], "Time":[]}
        sell_dict = {"ID":[], "Price":[], "Time":[]}

        # 第一张订单：买单
        t = dt.datetime(2024,9,1,10,30,1)
        new_order = {"Side":"Buy", "ID":1, "Price":100, "Time":t}

        # 期望：加入买簿，卖簿仍空
        expected_buy_dict  = {"ID":[1], "Price":[100], "Time":[t]}
        expected_sell_dict = {"ID":[], "Price":[], "Time":[]}

        new_buy_dict, new_sell_dict = priceTimeMatching1(buy_dict, sell_dict, new_order)

        self.assertDictEqual(new_buy_dict,  expected_buy_dict)
        self.assertDictEqual(new_sell_dict, expected_sell_dict)

    # —— 额外测试 2：对手盘为空的“卖单”对称场景 ——
    def test_priceTimeMatching1_noOpposingSellOrder_addsToDict(self):
        # 买簿为空；卖簿已有两张卖单（100, 100）
        buy_dict  = {"ID":[], "Price":[], "Time":[]}
        sell_dict = {"ID":[1,4], "Price":[100,100],
                     "Time":[dt.datetime(2024,9,1,10,30,3),
                             dt.datetime(2024,9,1,10,30,10)]}

        # 新来一张卖单（对手盘不存在 → 应直接加入卖簿，按价升序）
        new_order = {"Side":"Sell", "ID":6, "Price":101,
                     "Time":dt.datetime(2024,9,1,10,30,12)}

        expected_buy_dict  = {"ID":[], "Price":[], "Time":[]}
        expected_sell_dict = {"ID":[1,4,6], "Price":[100,100,101],
                              "Time":[dt.datetime(2024,9,1,10,30,3),
                                      dt.datetime(2024,9,1,10,30,10),
                                      dt.datetime(2024,9,1,10,30,12)]}

        new_buy_dict, new_sell_dict = priceTimeMatching1(buy_dict, sell_dict, new_order)

        self.assertDictEqual(new_buy_dict,  expected_buy_dict)
        self.assertDictEqual(new_sell_dict, expected_sell_dict)

    # —— 官方给的“可成交卖单”用例：应只撮合 1 张，然后停止 ——
    def test_priceTimeMatching1_matchableSellOrder_matchesOneOrder(self):
        buy_dict  = {"ID":[2,3], "Price":[99,98],
                     "Time":[dt.datetime(2024,9,1,10,30,1),
                             dt.datetime(2024,9,1,10,30,4)]}
        sell_dict = {"ID":[1,4], "Price":[100,100],
                     "Time":[dt.datetime(2024,9,1,10,30,3),
                             dt.datetime(2024,9,1,10,30,10)]}

        # 95 ≤ 买一(=99) → 可成交；应只把买簿的第 1 张移除
        new_order = {"Side":"Sell", "ID":8, "Price":95,
                     "Time":dt.datetime(2024,9,1,10,30,14)}

        expected_buy_dict  = {"ID":[3], "Price":[98],
                              "Time":[dt.datetime(2024,9,1,10,30,4)]}
        expected_sell_dict = {"ID":[1,4], "Price":[100,100],
                              "Time":[dt.datetime(2024,9,1,10,30,3),
                                      dt.datetime(2024,9,1,10,30,10)]}

        new_buy_dict, new_sell_dict = priceTimeMatching1(buy_dict, sell_dict, new_order)

        # 断言：只删除了对手盘的最优买单；卖簿保持不变
        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict,  expected_buy_dict)

    # —— 不可成交场景：新单应加入自己那一侧 ——
    def test_priceTimeMatching1_nonMatchingOrder_addsToDict(self):
        # 买簿两张（99, 98），卖簿两张（100, 100）
        buy_dict  = {"ID":[2,3], "Price":[99,98],
                     "Time":[dt.datetime(2024,9,1,10,30,1),
                             dt.datetime(2024,9,1,10,30,4)]}
        sell_dict = {"ID":[1,4], "Price":[100,100],
                     "Time":[dt.datetime(2024,9,1,10,30,3),
                             dt.datetime(2024,9,1,10,30,10)]}

        # 新来一张买单 97（97 < 卖一 100 → 不成交）
        new_order = {"Side":"Buy", "ID":9, "Price":97,
                     "Time":dt.datetime(2024,9,1,10,30,12)}

        # 期望：买簿追加到最后，卖簿不变
        expected_buy_dict  = {"ID":[2,3,9], "Price":[99,98,97],
                              "Time":[dt.datetime(2024,9,1,10,30,1),
                                      dt.datetime(2024,9,1,10,30,4),
                                      dt.datetime(2024,9,1,10,30,12)]}
        expected_sell_dict = {"ID":[1,4], "Price":[100,100],
                              "Time":[dt.datetime(2024,9,1,10,30,3),
                                      dt.datetime(2024,9,1,10,30,10)]}

        # 执行
        new_buy_dict, new_sell_dict = priceTimeMatching1(buy_dict, sell_dict, new_order)

        # 断言
        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict,  expected_buy_dict)


# %%
# Price time 1 suite:
def priceTimeMatching1Suite():
    loader = unittest.TestLoader()
    price_time_suite = loader.loadTestsFromTestCase(TestPriceTime1)
    return price_time_suite

runner = unittest.TextTestRunner(verbosity=1)
runner.run(priceTimeMatching1Suite())


# %%
order_list = [{"ID":1, "Side": "Sell", "Price": 101, "Time": dt.datetime(2024,9,1,10,30,1)},
              {"ID":2, "Side": "Buy", "Price":98, "Time": dt.datetime(2024,9,1,10,30,4)},
              {"ID":3, "Side": "Buy", "Price":99, "Time": dt.datetime(2024,9,1,10,30,10)},
              {"ID":4, "Side": "Sell", "Price": 101, "Time": dt.datetime(2024,9,1,10,30,14)}]
buy_dict = {"ID":[], "Price":[], "Time":[]}
sell_dict = {"ID":[], "Price":[], "Time":[]}

for order in order_list:
    buy_dict, sell_dict = priceTimeMatching1(buy_dict, sell_dict, order)

# It appears that the Buy and Sell dictionaries have formed correctly. We have 4 orders in the correct places.
print(buy_dict)
print(sell_dict)

# Can also visualise the LOB using the following:
buy_LOs = [buy_dict["Price"].count(i) for i in price_range]
sell_LOs = [sell_dict["Price"].count(i) for i in price_range]
plot_LOB(price_range, buy_LOs, sell_LOs).show()

# We make a function to do this for us in the future:
def plotLOBFromQuantityDicts(price_range,buy_dict, sell_dict):
    buy_LOs = [buy_dict["Price"].count(i) for i in price_range]
    sell_LOs = [sell_dict["Price"].count(i) for i in price_range]
    return plot_LOB(price_range, buy_LOs, sell_LOs)

# %%
# After order 5:
new_order_1 = {"ID":5, "Side":"Buy","Price":100, "Time": dt.datetime(2024,9,1,10,30,18)}
buy_dict,sell_dict = priceTimeMatching1(buy_dict, sell_dict, new_order_1)

plotLOBFromQuantityDicts(price_range, buy_dict, sell_dict).show()

# %%
# After order 6:
new_order_2 = {"ID":6, "Side":"Buy","Price":100, "Time": dt.datetime(2024,9,1,10,30,21)}
buy_dict,sell_dict = priceTimeMatching1(buy_dict, sell_dict, new_order_2)

plotLOBFromQuantityDicts(price_range, buy_dict, sell_dict).show()

# %%
# And lastly, after order 7:
new_order_3 = {"ID":7, "Side":"Sell","Price":99, "Time": dt.datetime(2024,9,1,10,30,23)}
buy_dict,sell_dict = priceTimeMatching1(buy_dict, sell_dict, new_order_3)

plotLOBFromQuantityDicts(price_range, buy_dict, sell_dict).show()

# %%
buy_dict

# %% [markdown]
# # 3.2.2- Price time with larger orders

# %%
# In the framework including quantities, our dictionaries look like:
Buy_dict = {"ID":[],"Price":[], "Quantity":[], "Time":[]}
Sell_dict = {"ID":[],"Price":[], "Quantity":[], "Time":[]}

# And a new order looks similar to:
new_order ={"ID":8, "Type":"Buy","Price":100, "Quantity":3,"Time": dt.datetime(2024,9,1,10,30,27)}

# %%
# Placeholder function:


class TestPriceTime(unittest.TestCase):
    def test_priceTimeMatching1_noOpposingBuyOrder_addsToDict(self):
        buy_dict = {"ID":[2,3], 
                    "Price":[99,98], 
                    "Time":[dt.datetime(2024,9,1,10,30,1), dt.datetime(2024,9,1,10,30,4)],
                    "Quantity":[1,3]}
        sell_dict = {"ID":[], "Price":[], "Time":[], "Quantity":[]}

        new_order = {"Side":"Buy",
                      "ID":5, 
                      "Price":100,
                      "Time":dt.datetime(2024,9,1,10,30,10),
                      "Quantity":3}

        expected_buy_dict = {"ID":[5,2,3], 
                             "Price":[100,99,98], 
                             "Time":[dt.datetime(2024,9,1,10,30,10),dt.datetime(2024,9,1,10,30,1), dt.datetime(2024,9,1,10,30,4)],
                             "Quantity":[3,1,3]}
        expected_sell_dict = {"ID":[], "Price":[], "Time":[], "Quantity":[]}

        new_buy_dict, new_sell_dict = priceTimeMatching(buy_dict, sell_dict, new_order)

        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict, expected_buy_dict)
    
    def test_priceTimeMatching1_noOpposingSellOrder_addsToDict(self):
        buy_dict = {"ID":[], "Price":[], "Time":[], "Quantity":[]}
        sell_dict = {"ID":[1,4], 
                     "Price":[100,100],
                      "Time":[dt.datetime(2024,9,1,10,30,3), dt.datetime(2024,9,1,10,30,10)],
                      "Quantity":[4,7]}

        new_order = {"Side":"Sell",
                     "ID":6, 
                     "Price":102, 
                     "Time":dt.datetime(2024,9,1,10,30,11),
                     "Quantity":2}

        expected_buy_dict = {"ID":[], "Price":[], "Time":[], "Quantity":[]}
        expected_sell_dict = {"ID":[1,4,6],
                              "Price":[100,100,102], 
                              "Time":[dt.datetime(2024,9,1,10,30,3), dt.datetime(2024,9,1,10,30,10),dt.datetime(2024,9,1,10,30,11)],
                              "Quantity":[4,7,2]}

        new_buy_dict, new_sell_dict = priceTimeMatching(buy_dict, sell_dict, new_order)

        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict, expected_buy_dict)
    
    def test_priceTimeMatching1_matchableBuyOrder_matchesOneOrder(self):
        buy_dict = {"ID":[2,3],
                    "Price":[99,98], 
                    "Time":[dt.datetime(2024,9,1,10,30,1), dt.datetime(2024,9,1,10,30,4)],
                    "Quantity":[3,5]}
        sell_dict = {"ID":[1,4], 
                     "Price":[100,100],
                      "Time":[dt.datetime(2024,9,1,10,30,3), dt.datetime(2024,9,1,10,30,10)],
                      "Quantity":[4,7]}
        
        new_order = {"Side":"Buy", 
                     "ID":7, 
                     "Price":100, 
                     "Time":dt.datetime(2024,9,1,10,30,11),
                     "Quantity":6} 
        # 6 < 4+7 so it should partially fill the second order.
        # Checks if removing some of the quantity is working.
        expected_buy_dict = {"ID":[2,3],
                    "Price":[99,98], 
                    "Time":[dt.datetime(2024,9,1,10,30,1), dt.datetime(2024,9,1,10,30,4)],
                    "Quantity":[3,5]}
        expected_sell_dict = {"ID":[4], 
                     "Price":[100],
                      "Time":[dt.datetime(2024,9,1,10,30,10)],
                      "Quantity":[5]}

        new_buy_dict, new_sell_dict = priceTimeMatching(buy_dict, sell_dict, new_order)

        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict, expected_buy_dict)

    def test_priceTimeMatching1_matchableSellOrder_matchesOneOrder(self):
        buy_dict = {"ID":[2,3],
                    "Price":[99,98], 
                    "Time":[dt.datetime(2024,9,1,10,30,1), dt.datetime(2024,9,1,10,30,4)],
                    "Quantity":[3,5]}
        sell_dict = {"ID":[1,4], 
                     "Price":[100,100],
                      "Time":[dt.datetime(2024,9,1,10,30,3), dt.datetime(2024,9,1,10,30,10)],
                      "Quantity":[4,7]}

        new_order = {"Side":"Sell",
                     "ID":8, 
                     "Price":95, 
                     "Time":dt.datetime(2024,9,1,10,30,14),
                     "Quantity":9}
        
        # 9 > 3+5 so it should wipe out both buy_dict orders and then become a sell order:
        expected_buy_dict = {"ID":[], "Price":[], "Time":[], "Quantity":[]}
        expected_sell_dict = {"ID":[8,1,4], "Price":[95,100,100], 
                              "Time":[dt.datetime(2024,9,1,10,30,14),dt.datetime(2024,9,1,10,30,3), dt.datetime(2024,9,1,10,30,10)],
                              "Quantity":[1,4,7]}

        new_buy_dict, new_sell_dict = priceTimeMatching(buy_dict, sell_dict, new_order)

        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict, expected_buy_dict)

# Test suite for the new version of Price Time matching
def priceTimeMatchingSuite():

    loader = unittest.TestLoader()
    price_time_suite = loader.loadTestsFromTestCase(TestPriceTime)
    return price_time_suite

# Running the tests, we see every test fails.
runner.run(priceTimeMatchingSuite())

# %%
# Updated algorithm:
def priceTimeMatching(Buy_dict, Sell_dict, new_order):
    order_side = new_order["Side"]          # 新单方向："Buy" 或 "Sell"
    quantity   = new_order["Quantity"]      # 新单要成交的总数量
    # 根据方向把（买簿, 卖簿）重排成（同侧簿, 对手簿），这样后面就不用分别写两套逻辑
    same_side_dict, opposing_dict = swapDictSides(Buy_dict, Sell_dict, order_side)

    best_order_index = 0                    # 约定最优价总在索引 0（买簿降序、卖簿升序时成立）

    # —— 主循环：只要新单还有剩余数量，且对手盘不为空，就尝试撮合 ——
    while quantity > 0 and len(opposing_dict["ID"]) > 0:
        best_price = opposing_dict["Price"][best_order_index]    # 取对手盘的最优价
        # 判断是否“跨越价差”（买：新单价 ≥ 卖一；卖：新单价 ≤ 买一），不跨越就停止
        if not checkCrossesSpread(new_order["Price"], best_price, order_side):
            break

        # 取出最优那张对手盘订单的数量
        opp_qty = opposing_dict["Quantity"][best_order_index]

        if quantity >= opp_qty:
            # 情形 A：新单数量 ≥ 对手单数量 → 吃掉整张对手单
            quantity -= opp_qty                            # 新单剩余数量减少
            deleteRow(opposing_dict, best_order_index)     # 对手单被完全成交，从簿里删除
            # 删除后下一张最优单会自动顶到索引 0，下一轮 while 继续处理
        else:
            # 情形 B：新单数量 < 对手单数量 → 只部分成交
            opposing_dict["Quantity"][best_order_index] = opp_qty - quantity  # 修改对手单剩余数量
            quantity = 0                                   # 新单被完全吃完
            break                                          # 结束循环

    # 循环结束后：如果新单还有剩余（说明对手盘吃不动或已空），把剩余部分挂到同侧簿
    if quantity > 0:
        # 构造“剩余新单”的行（同侧簿包含的列必须都提供：ID/Price/Time/Quantity）
        residual_row = {
            "ID":       new_order["ID"],
            "Price":    new_order["Price"],
            "Time":     new_order["Time"],
            "Quantity": quantity,
        }
        # 按“买：价降序/时升序；卖：价升序/时升序”的规则插入同侧簿
        addToDict(same_side_dict, residual_row, order_side)

    # 返回时把（同侧簿, 对手簿）顺序换回（固定返回 buy_dict, sell_dict 的顺序）
    return swapDictSides(same_side_dict, opposing_dict, order_side)


# %%
runner.run(priceTimeMatchingSuite())

# %% [markdown]
# ### 3.2.3 - Cancellations

# %%
def cancelOrder(buy_dict, sell_dict, order_ID):
    buy_IDs = buy_dict["ID"]
    sell_IDs = sell_dict["ID"]
    if order_ID in buy_IDs:
        order_idx = buy_IDs.index(order_ID)
        deleteRow(buy_dict, order_idx)
    elif order_ID in sell_IDs:
        order_idx = sell_IDs.index(order_ID)
        deleteRow(sell_dict, order_idx)
    else:
        print(f"Error: order {order_ID} already matched, cannot cancel.")
    return(buy_dict, sell_dict)

# %%
# Checking what happens when removing an order currently in the LOB:
buy_dict= {"ID":[2,3],
           "Price":[99,98], 
           "Time":[dt.datetime(2024,9,1,10,30,1), dt.datetime(2024,9,1,10,30,4)],
           "Quantity":[3,5]}
sell_dict = {"ID":[1,4], 
             "Price":[100,100],
             "Time":[dt.datetime(2024,9,1,10,30,3), dt.datetime(2024,9,1,10,30,10)],
             "Quantity":[4,7]}

buy_dict, sell_dict = cancelOrder(buy_dict, sell_dict, 3)
# 
# Nothing changes in Buy_dict, Sell_dict as there is no order to cancel. Also prints an error as intended.

# Removing a non existent order prints an error message and doesn't change the LOB.
buy_dict, sell_dict = cancelOrder(buy_dict, sell_dict, 7)

plotLOBFromQuantityDicts(price_range, buy_dict, sell_dict).show()

# %% [markdown]
# # 3.2.4 - Pro Rata matching algorithm

# %%
class TestProRata(unittest.TestCase):
    def test_proRata_matchableBuyOrder_matchesProportionally(self):
        # Testing that a buy order is matched until it matches
        # proportioal to the quantities in the last layer.
        buy_dict = {"ID":[7], "Price":[98], "Quantity":[10], "Time":[dt.datetime(2024,9,1,10,42,0)]}
        sell_dict = {"ID":[5,2,3],
                     "Price":[99,100,100],
                     "Quantity":[20,30,60],
                     "Time":[dt.datetime(2024,9,1,10,40,0),dt.datetime(2024,9,1,10,36,0), dt.datetime(2024,9,1,10,37,0)]}
        new_order = {"Side":"Buy", "ID":8, "Price":100, "Quantity":56, "Time":dt.datetime(2024,9,1,10,43,0)}
        # Should match 20 with the $99 orders, leaving 36.
        # 36 / (30+60) * 30 = 12
        # 36 / (30+60) * 60 = 24
        # So after matching with the $100 orders, we get:
        # 30 - 12 = 18 shares in order 3
        # 60 - 24 = 36 shares in order 5.
        # (And no change to the buy dictionary)
        expected_buy_dict = {"ID":[7], "Price":[98], "Quantity":[10], "Time":[dt.datetime(2024,9,1,10,42,0)]}
        expected_sell_dict = {"ID":[2,3],
                     "Price":[100,100],
                     "Quantity":[18,36],
                     "Time":[dt.datetime(2024,9,1,10,36,0), dt.datetime(2024,9,1,10,37,0)]}

        # Computing our algorithm's results:
        buy_dict, sell_dict = proRataMatching(buy_dict, sell_dict, new_order)

        self.assertDictEqual(expected_buy_dict, buy_dict)
        self.assertDictEqual(expected_sell_dict, sell_dict)

    def test_proRata_partialMatchableBuyOrder_matchThenAdd(self):
        # Testing that a buy order is matched until it fulfills all
        # sell orders until it's added to the LOB.
        buy_dict = {"ID":[7], "Price":[98], "Quantity":[10], "Time":[dt.datetime(2024,9,1,10,42,0)]}
        sell_dict = {"ID":[5,2,3],
                     "Price":[99,100,100],
                     "Quantity":[20,30,60],
                     "Time":[dt.datetime(2024,9,1,10,40,0),dt.datetime(2024,9,1,10,36,0), dt.datetime(2024,9,1,10,37,0)]}
        
        new_order = {"Side":"Buy", "ID":8, "Price":99, "Quantity":56, "Time":dt.datetime(2024,9,1,10,43,0)}

        # Price is 99 so it matches with order 5 until order 5 is exhausted,
        # then is added to the buy dict.

        expected_buy_dict = {"ID":[8,7], "Price":[99,98], "Quantity":[36,10], 
                             "Time":[dt.datetime(2024,9,1,10,43,0),dt.datetime(2024,9,1,10,42,0)]}
        expected_sell_dict = {"ID":[2,3],"Price":[100,100],"Quantity":[30,60],
                              "Time":[dt.datetime(2024,9,1,10,36,0), dt.datetime(2024,9,1,10,37,0)]}
        
        buy_dict, sell_dict = proRataMatching(buy_dict, sell_dict, new_order)

        self.assertDictEqual(expected_buy_dict, buy_dict)
        self.assertDictEqual(expected_sell_dict, sell_dict)
        
    def test_proRata_order15_sell98_qty40(self):
        # 订单 15：卖 40，价 98；先吃 99 档 10 股，剩余 30 对 98 档(30/40/80)按比例分配
        buy_dict = {
            "ID":[11,7,9,14],
            "Price":[99,98,98,98],
            "Quantity":[10,30,40,80],
            "Time":[dt.datetime(2024,9,1,10,44,0),
                    dt.datetime(2024,9,1,10,42,0),
                    dt.datetime(2024,9,1,10,44,0),
                    dt.datetime(2024,9,1,10,47,0)]
        }
        sell_dict = {"ID":[], "Price":[], "Quantity":[], "Time":[]}
        new_order = {"Side":"Sell","ID":15,"Price":98,
                     "Quantity":40,"Time":dt.datetime(2024,9,1,10,49,0)}

        expected_buy = {
            "ID":[7,9,14],
            "Price":[98,98,98],
            "Quantity":[24,32,64],
            "Time":[dt.datetime(2024,9,1,10,42,0),
                    dt.datetime(2024,9,1,10,44,0),
                    dt.datetime(2024,9,1,10,47,0)]
        }
        expected_sell = {"ID":[], "Price":[], "Quantity":[], "Time":[]}

        out_buy, out_sell = proRataMatching(buy_dict, sell_dict, new_order)
        self.assertDictEqual(out_buy, expected_buy)
        self.assertDictEqual(out_sell, expected_sell)

    def test_proRata_order16_sell97_qty22(self):
        # 订单 16：卖 22，价 97；对 98 档(24,32,64)做 Pro Rata 后余 2 股 FIFO 给最老的 7
        buy_dict = {
            "ID":[7,9,14],
            "Price":[98,98,98],
            "Quantity":[24,32,64],
            "Time":[dt.datetime(2024,9,1,10,42,0),
                    dt.datetime(2024,9,1,10,44,0),
                    dt.datetime(2024,9,1,10,47,0)]
        }
        sell_dict = {"ID":[], "Price":[], "Quantity":[], "Time":[]}
        new_order = {"Side":"Sell","ID":16,"Price":97,
                     "Quantity":22,"Time":dt.datetime(2024,9,1,10,52,0)}

        expected_buy = {
            "ID":[7,9,14],
            "Price":[98,98,98],
            "Quantity":[18,27,53],
            "Time":[dt.datetime(2024,9,1,10,42,0),
                    dt.datetime(2024,9,1,10,44,0),
                    dt.datetime(2024,9,1,10,47,0)]
        }
        expected_sell = {"ID":[], "Price":[], "Quantity":[], "Time":[]}

        out_buy, out_sell = proRataMatching(buy_dict, sell_dict, new_order)
        self.assertDictEqual(out_buy, expected_buy)
        self.assertDictEqual(out_sell, expected_sell)



# %%
def proRataMatchingSuite():

    loader = unittest.TestLoader()
    price_time_suite = loader.loadTestsFromTestCase(TestProRata)
    return price_time_suite

# Running the tests, we see every test fails.
runner.run(proRataMatchingSuite())

# %%
def priceTimeOneStep(same_side_dict, opposing_dict, new_order):
    best_order_index = 0                           # 对手盘“最优价”永远在索引 0
    quantity = new_order["Quantity"]               # 本轮剩余需要撮合的数量
    while quantity > 0:                            # 只要还有量，就尝试继续
        # 先检查是否还能与对手盘最优价成交（跨越价差？）
        same_side_dict, opposing_dict, _, do_matching = \
            checkIfMatch(same_side_dict, opposing_dict, new_order)
        if not do_matching:                        # 不能成交就停止，返回当前数量
            return (same_side_dict, opposing_dict, quantity)

        best_order_quantity = opposing_dict["Quantity"][best_order_index]  # 对手最优单的数量
        if quantity >= best_order_quantity:        # 新单量 ≥ 对手单 → 吃掉整张对手单
            deleteRow(opposing_dict, best_order_index)   # 从对手簿删除
            quantity = quantity - best_order_quantity     # 更新新单剩余量
            new_order["Quantity"] = quantity              # 写回，便于后续继续循环
        else:
            # 新单量 < 对手单 → 仅部分成交：对手单数量减少，新单耗尽
            opposing_dict["Quantity"][best_order_index] = best_order_quantity - quantity
            quantity = 0
            new_order["Quantity"] = 0
    return (same_side_dict, opposing_dict, quantity)      # 返回处理后字典和剩余量（0 或正数）


def proRataOneStep(same_side_dict, opposing_dict, new_order):
    best_order_index = 0                               # 最优价从索引 0 开始
    opposing_prices = opposing_dict["Price"]           # 对手盘价格列表（买簿是降序，卖簿是升序）
    best_opposing_price = opposing_prices[best_order_index]  # 取当下最优价
    order_side = new_order["Side"]                     # 新单方向

    # 找到“与最优价相同”的一整段区间 [0, newest_best_order_index)
    if order_side == "Sell":
        # 对手盘为买簿（降序），翻转成升序再用 bisect_left 得到“左边界”
        opposing_prices.reverse()
        rev_left = bisect.bisect_left(opposing_prices, best_opposing_price)
        newest_best_order_index = len(opposing_prices) - rev_left
        opposing_prices.reverse()
    else:
        # 对手盘为卖簿（升序），直接用 bisect_right 得到“右边界”
        newest_best_order_index = bisect.bisect_right(opposing_prices, best_opposing_price)

    # 取出这段最优价上的所有数量
    best_order_quantities = opposing_dict["Quantity"][best_order_index:newest_best_order_index]
    total_quantity = sum(best_order_quantities)        # 该价位总量
    quantity = new_order["Quantity"]                   # 新单当前剩余量

    if quantity >= total_quantity:
        # 情况 A：新单量 ≥ 该价位总量 → 整段价位全部吃掉
        for _ in range(best_order_index, newest_best_order_index):
            deleteRow(opposing_dict, best_order_index) # 连续删除索引 0（每删一次后续前移）
        quantity = quantity - total_quantity           # 更新新单剩余量
        new_order["Quantity"] = quantity               # 写回供外层循环继续
        return (same_side_dict, opposing_dict, new_order, quantity)
    else:
        # 情况 B：新单量 < 该价位总量 → 按比例吃（下取整），余数用 FIFO
        # 计算按比例应该吃的“整数份额”
        ratio = np.array(best_order_quantities, dtype=float) / float(total_quantity)
        traded = np.floor(ratio * quantity).astype(int)          # 各订单成交量（取整）
        remaining = (np.array(best_order_quantities) - traded).astype(int)  # 各订单剩余量

        # 新单剩余（是“未分配的余数”，下一步用 FIFO）
        quantity = int(quantity - int(traded.sum()))

        # 把该价位区间的数量更新为“剩余量”
        opposing_dict["Quantity"][best_order_index:newest_best_order_index] = remaining.tolist()

        # 用 Price–Time 规则对“余数”进行补充撮合（FIFO）
        new_order["Quantity"] = quantity
        same_side_dict, opposing_dict, quantity = priceTimeOneStep(same_side_dict, opposing_dict, new_order)

        # 这里 new_order 的内容已不再需要（数量已用尽或外层将继续），置空仅作语义提示
        new_order = None
        return (same_side_dict, opposing_dict, new_order, quantity)


def proRataMatching(Buy_dict, Sell_dict, new_order):
    order_side = new_order["Side"]                                  # 新单方向
    quantity = new_order["Quantity"]                                 # 新单剩余量
    # 交换成（同侧簿, 对手簿），便于复用统一逻辑
    same_side_dict, opposing_dict = swapDictSides(Buy_dict, Sell_dict, order_side)

    # 只要新单还有量，就继续尝试匹配
    while quantity > 0:
        # 先看是否还能跨越价差；若不能，checkIfMatch 会负责把新单加入同侧簿
        same_side_dict, opposing_dict, new_order, do_matching = \
            checkIfMatch(same_side_dict, opposing_dict, new_order)
        if not do_matching:
            # 不能成交 → 直接返回（并把字典顺序换回 buy/sell）
            return swapDictSides(same_side_dict, opposing_dict, order_side)

        # 能成交 → 进行一轮 Pro Rata；函数会更新字典并返回新的剩余量
        same_side_dict, opposing_dict, new_order, quantity = \
            proRataOneStep(same_side_dict, opposing_dict, new_order)

    # 新单数量耗尽：把顺序换回（buy_dict, sell_dict）并返回
    return swapDictSides(same_side_dict, opposing_dict, order_side)


# %%
runner.run(proRataMatchingSuite())

# %% [markdown]
# # 3.2.5 - Top Orders and LMMs

# %% [markdown]
# # 3.3 - An OOP implementation

# %% [markdown]
# ### 3.3.1 - Limit order book sides in OOP

# %%
# Firstly, we'll implement the LOB dictionaries as a class:

class LOBSide():
    def __init__(self, side, init_dict, sort_1 = "Price", sort_2 = "Time"):
        self.side = side
        self.orders = init_dict
        self.sort_1 = sort_1
        self.sort_2 = sort_2

    # This class has methods resembling the helper functions we made earlier with some minor tweaks. 
    # - We access the half of the LOB with self.orders
    # - getSortIdx can be simplified as some cases are unnecessary
    # - addOrder is significantly simpler
    # - We've changed how we implemented some other tasks, such as getOrderIdxByID and cancelOrderByID.
    # - We don't have swap dict sides in this part of the implementation.
    # - We've added two new funtions, 
    #   - modifyOrder (which wil be used to change the top amount and quantity of existing orders)
    #   - getBestOrderQuantities (which will be useful for Pro Rata matching)

    def insertOrderInPosition(self, new_order, index):
        # Check if the index is valid:
        n_orders = len(self.orders["ID"])
        if index >= n_orders + 1 or index < 0:
            raise ValueError("Index not in range")
        # Check if the order has the required fields
        fields_in_LOB = set(self.orders.keys())
        fields_in_new_order = set(new_order.keys())

        if not fields_in_LOB <= fields_in_new_order:
            raise TypeError("New row is missing a required field")
        # Adding the order
        for key in self.orders:
            self.orders[key].insert(index, new_order[key])
        # No need to return the value.
    
    def getSortIdx(self, new_row):
        sort_1_list = self.orders[self.sort_1]
        n_rows = len(sort_1_list)
        # Gets the first part of the insertion index, reversing the order if it's a Buy dictionary as we
        # want the lowest prices first in that case. 
        # We no longer need to do the case where we reverse the order for the secondary
        # sorting variable because this isn't needed for any of our matching algorithms
        if self.side == "Buy":
            sort_1_list.reverse()
            rev_lower_index = bisect.bisect_left(sort_1_list, new_row[self.sort_1])
            rev_upper_index = bisect.bisect_right(sort_1_list, new_row[self.sort_1])
            sort_1_list.reverse()
            upper_index = n_rows - rev_lower_index
            lower_index = n_rows - rev_upper_index
        else:
            lower_index = bisect.bisect_left(sort_1_list, new_row[self.sort_1])
            upper_index = bisect.bisect_right(sort_1_list, new_row[self.sort_1])
        if self.sort_2 is None:
            return upper_index
        # Gets the secord part of the index (when sorted by Time, typically)
        sort_2_list = self.orders[self.sort_2][lower_index:upper_index]
        sub_index = bisect.bisect_right(sort_2_list, new_row[self.sort_2])
        # Returns the final position.
        return(lower_index + sub_index)
    
    def addOrder(self, new_order):
        # Gets the index then adds the order to the LOB
        new_idx = self.getSortIdx(new_order)
        self.insertOrderInPosition(new_order, new_idx)
    
    def deleteOrder(self, index):
        n_orders = len(self.orders["ID"])
        # Checks if order exists
        if index >= n_orders or index < 0:
            raise ValueError("Index not in range")
        # Removes that order from our dictionary.
        for key in self.orders:
            self.orders[key].pop(index)
    
    def getOrder(self, index):
        n_orders = len(self.orders["ID"])
        # Checks if order exists
        if index >= n_orders or index < 0:
            raise ValueError("Index not in range")
        output_order = {}
        for key in self.orders:
            output_order[key] = self.orders[key][index]
        return output_order
     
    def getOrderIdxByID(self,ID):
        if ID in self.orders["ID"]:
            index = self.orders["ID"].index(ID)
            return index
        else:
            return None
        
    def cancelOrderByID(self, ID):
        index = self.getOrderIdxByID(ID)
        if index is None:
            return None
        else:
            self.deleteOrder(index)

    def checkCrossesSpread(self,order_price):
        # If there are no orders, no one can cross the spread.
        if len(self.orders["ID"]) == 0:
            return False
        
        # Otherwise, we check if the prices match up:
        if self.side == "Buy":
            return order_price <= self.orders["Price"][0]
        else:
            return order_price >= self.orders["Price"][0]
    
    def modifyOrder(self, index, key, new_value):
        self.orders[key][index] = new_value
    
    def getBestOrderQuantities(self):
        # If there are no orders, no quantities exist
        if len(self.orders["ID"]) == 0:
            return None
        # Getting the prices and determining the upper and lower indices for the prices equal to the best one.
        prices = self.orders["Price"]
        best_price = prices[0]
        lower_index = 0
        # Find the last index with the same price value as the first one
        n_rows = len(prices)
        if self.side == "Buy":
            prices.reverse()
            rev_lower_index = bisect.bisect_left(prices, best_price)
            prices.reverse()
            upper_index = n_rows - rev_lower_index
        else:
            upper_index = bisect.bisect_right(prices, best_price)  

        # Get the quantity list:
        quantities = self.orders["Quantity"][lower_index:upper_index]
        return quantities
    
# Now, a lazy way to check our new implementations pass the tests for the helper functions is to 
# re define the functions to make use of our methods. 

# It would be better, in a production setting, to refactor the tests to reflect the OOP nature, but for
# brevity we'll be using this shortcut.

def insertRowInPosition(side_dict, new_row, index):
    LOB_side = LOBSide(side = "Buy", init_dict = side_dict)
    LOB_side.insertOrderInPosition(new_row, index)
    return LOB_side.orders

def getSortIdx(side_dict, new_row, sort_1, sort_2 = None, reverse_1 = False, reverse_2 = False):
    """
    作用：复用旧测试的签名，内部用 LOBSide 的 getSortIdx 来计算插入位置。
    关键点：
    - 旧接口里的 reverse_1 表示“主键需要降序”（买侧的价格排序），
      而 LOBSide 用 self.side == "Buy" 来触发展开“价格降序”的逻辑。
      因此：reverse_1=True → side='Buy'；否则 → side='Sell'。
    - sort_1/sort_2 直接传给 LOBSide，使其按同样的主次键排序。
    """
    # 根据 reverse_1 决定在 LOBSide 中使用哪一侧（影响主键升降序）
    side_for_lob = "Buy" if reverse_1 else "Sell"

    # 创建一个 LOBSide 实例；init_dict 使用传入的 side_dict，不复制即可（类方法不会修改它）
    lob = LOBSide(side=side_for_lob, init_dict=side_dict, sort_1=sort_1, sort_2=sort_2)

    # 调用类方法计算应插入的位置并返回
    return lob.getSortIdx(new_row)


def deleteRow(side_dict, index):
    """
    作用：复用旧测试的签名，内部用 LOBSide.deleteOrder 删除一行。
    说明：deleteOrder 不依赖 side 的方向，这里随便给个值即可（用 'Buy'）。
    """
    # 创建 LOBSide 实例（方向不影响删除行为）
    lob = LOBSide(side="Buy", init_dict=side_dict)

    # 调用类方法删除 index 行（会同步删所有列的该行）
    lob.deleteOrder(index)

    # 返回更新后的字典（与旧函数保持一致）
    return lob.orders


def addToDict(side_dict, new_order, order_side):
    """
    作用：按“买：价格降序、时间升序；卖：价格升序、时间升序”的规则插入新订单。
    做法：把 order_side 直接作为 LOBSide 的 side，让类内部按侧别决定升降序。
    """
    # 创建对应侧的 LOBSide（order_side 为 "Buy" 或 "Sell"）
    lob = LOBSide(side=order_side, init_dict=side_dict)

    # 按类的方法插入新订单（内部会先算位置再插入）
    lob.addOrder(new_order)

    # 返回更新后的字典，保持与旧接口一致
    return lob.orders


def checkCrossesSpread(order_price, best_price, order_side):
    """
    作用：判断新单价格是否“跨越价差”。
    为了复用类方法，我们构造一个仅含“最优价”的临时 LOBSide（对手盘一侧），
    然后调用 LOBSide.checkCrossesSpread(order_price)。
    说明：
    - 买单应与“卖侧最优价”比较：order_price >= best_ask
    - 卖单应与“买侧最优价”比较：order_price <= best_bid
    - best_price 在这里就是对手盘的最优价（由调用方提供）
    """
    # 先确定“对手盘”的侧别：买单对应卖侧；卖单对应买侧
    opposing_side = "Sell" if order_side == "Buy" else "Buy"

    # 如果对手盘根本没有价格（防御性判断），则无法成交
    if best_price is None:
        return False

    # 构造一个“只有最优价一行”的最小字典，满足 LOBSide 的字段需求
    tmp_dict = {
        "ID":       [0],                       # 占位 ID
        "Price":    [best_price],              # 对手盘最优价放在第一行
        "Time":     [dt.datetime(1970,1,1)],   # 任意时间占位（不会被用到）
        "Quantity": [1],                       # 数量占位（同样不会被用到）
    }

    # 创建对手盘 LOB 对象
    opp_lob = LOBSide(side=opposing_side, init_dict=tmp_dict)

    # 用类方法判断是否“跨越价差”，并返回 True/False
    return opp_lob.checkCrossesSpread(order_price)




# ------

# We make no changes to swapDictSides, as it's no longer really in our helper function set.

def swapDictSides(buy_dict, sell_dict, order_side):
    if order_side == "Buy":
        return (buy_dict, sell_dict)
    elif order_side == "Sell":
        return(sell_dict, buy_dict)


runner.run(helperFunctionSuite())

# %% [markdown]
# ### 3.3.2 - The Base matching algorithm class

# %%
class BaseMatchingAlgorithm():
    def __init__(self, LOB_buy, LOB_sell):
        # Store the LOB components.
        LOB_dict = {"Buy":LOB_buy, "Sell":LOB_sell}
        self.LOB_dict = LOB_dict
    
    # Placeholder method that uses FIFO matching for now. Other classes will inherit most methods
    # from this class be able to change this behaviour.
    def matchNewOrder(self,new_order):
        self.priceTimeMatchOrder(new_order)
    
    def getLOBDicts(self):
        return(self.LOB_dict["Buy"].orders, self.LOB_dict["Sell"].orders)
    
    def addOpposingSide(self,new_order):
        # Adding the opposing side data to new orders.
        if new_order["Side"] == "Buy":
            new_order["OpposingSide"] = "Sell"
        else:
            new_order["OpposingSide"] = "Buy"
        return new_order
    
    def deleteOpposingOrder(self, new_order, LOB_index = 0):
        # Deletes the order opposing the new one (defaults to the best order).
        self.LOB_dict[new_order["OpposingSide"]].deleteOrder(LOB_index)
    
    def processNewOrder(self, new_order):
        # We'll call this method to add a new order to the LOB / match it.
        new_order = self.addOpposingSide(new_order)
        self.matchNewOrder(new_order)

    def checkIfMatch(self, new_order):
        # Check if an order can be matched
        matchable = self.LOB_dict[new_order["OpposingSide"]].checkCrossesSpread(new_order["Price"])
        return matchable

    def insertOrderIntoLOB(self, new_order):
        # Insert a new order when it cannot be matched
        self.LOB_dict[new_order["Side"]].addOrder(new_order)
    
    def cancelOrder(self, order_ID):
        self.LOB_dict["Buy"].cancelOrderByID(order_ID)
        self.LOB_dict["Sell"].cancelOrderByID(order_ID)
    
    def priceTimeOneStep(self, new_order):
        # We may need to perform price time matching, but only for a single step.
        # (For instance, if using LMMs with price time matching and no Pro Rata matching).
        # We can assume matchability has already been checked.
        # Also returns whether the current order is finished / fully matched or not.
        quantity = new_order["Quantity"]
        best_order_quantities = self.LOB_dict[new_order["OpposingSide"]].getBestOrderQuantities()
        best_order_quantity = best_order_quantities[0]
        finished = False
        if quantity >= best_order_quantity:
            quantity = quantity - best_order_quantity
            new_order["Quantity"] = quantity
            self.deleteOpposingOrder(new_order)
            matchable = self.checkIfMatch(new_order)
            if quantity == 0:
                finished = True
            elif not matchable:
                self.insertOrderIntoLOB(new_order)
                finished = True
        else:
            new_best_order_quantity = best_order_quantity - quantity
            self.LOB_dict[new_order["OpposingSide"]].modifyOrder(0, "Quantity", new_best_order_quantity)
            new_order["Quantity"] =0
            finished = True
        return new_order, finished
    
    def priceTimeMatchOrder(self, new_order):
        quantity = new_order["Quantity"]
        while quantity > 0:
            matchable = self.checkIfMatch(new_order)
            # If we can't match it with anything, we add it into our LOB
            if not matchable:
                self.insertOrderIntoLOB(new_order)
                return
            # Otherwise, we match it with the opposing trades:
            new_order, finished = self.priceTimeOneStep(new_order)
            if finished:
                return new_order, finished
            quantity = new_order["Quantity"]

# This matching base class should have everything we need to pass the Price Time matching algorithm unit tests. Once again,
# these tests need some modification (and in practice should probably be rewritten). 

# This time we only need to rewrite the priceTimeMatching function:
def priceTimeMatching(buy_dict, sell_dict, new_order):
    LOB_buy = LOBSide(side = "Buy", init_dict = buy_dict)
    LOB_sell = LOBSide(side = "Sell", init_dict = sell_dict)
    priceTimeMatchingAlgo = BaseMatchingAlgorithm(LOB_buy, LOB_sell)
    priceTimeMatchingAlgo.processNewOrder(new_order)
    LOB_dict = priceTimeMatchingAlgo.LOB_dict
    new_LOB_buy = LOB_dict["Buy"].orders
    new_LOB_sell = LOB_dict["Sell"].orders
    return new_LOB_buy, new_LOB_sell

runner.run(priceTimeMatchingSuite())

# %% [markdown]
# ### 3.3.3 - Top orders and LMM Priority

# %%
def GetFullMatching(buy_dict, sell_dict, matching_rules):
    LOB_buy = LOBSide(side = "Buy", init_dict = buy_dict)
    LOB_sell = LOBSide(side = "Sell", init_dict = sell_dict)
    FullMatchingAlgo = TopLMMProRataAlgorithm(LOB_buy, LOB_sell, matching_rules)
    return FullMatchingAlgo

class TestTopLMMProRataClass(unittest.TestCase):
    # Some tests for error raising:

    def test_TopLMMProRataAlgorithm_unsupportedMatchingType_ValueError(self):
        buy_dict = {"ID":[7], "Price":[98], "Quantity":[10], "Time":[dt.datetime(2024,9,1,10,42,0)]}
        sell_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}
        matching_rules = "FIFo"
        expected_exception = ValueError
        self.assertRaises(expected_exception, GetFullMatching, buy_dict, sell_dict, matching_rules)

    def test_TopLMMProRataAlgorithm_noBaseMatchingType_ValueError(self):
        buy_dict = {"ID":[7], "Price":[98], "Quantity":[10], "Time":[dt.datetime(2024,9,1,10,42,0)]}
        sell_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}
        matching_rules = "LMM+Top"
        expected_exception = ValueError
        self.assertRaises(expected_exception, GetFullMatching, buy_dict, sell_dict, matching_rules)

    def test_TopLMMProRataAlgorithm_twoBaseMatchingType_ValueError(self):
        buy_dict = {"ID":[7], "Price":[98], "Quantity":[10], "Time":[dt.datetime(2024,9,1,10,42,0)]}
        sell_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}
        matching_rules = "FIFO+ProRata"
        expected_exception = ValueError
        self.assertRaises(expected_exception, GetFullMatching, buy_dict, sell_dict, matching_rules)
    
    def test_TopLMMProRataAlgorithm_topUnsupportedByLOBMatching_TypeError(self):
        buy_dict = {"ID":[7], "Price":[98], "Quantity":[10], "Time":[dt.datetime(2024,9,1,10,42,0)]}
        sell_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}
        matching_rules = "Top+FIFO"
        expected_exception = TypeError
        self.assertRaises(expected_exception, GetFullMatching, buy_dict, sell_dict, matching_rules)
    
    def test_TopLMMProRataAlgorithm_LMMUnsupportedByLOBMatching_TypeError(self):
        buy_dict = {"ID":[7], "Price":[98], "Quantity":[10], "Time":[dt.datetime(2024,9,1,10,42,0)]}
        sell_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}
        matching_rules = "LMM+FIFO"
        expected_exception = TypeError
        self.assertRaises(expected_exception, GetFullMatching, buy_dict, sell_dict, matching_rules)
    
    def test_TopLMMProRataAlgorithm_LMMPropUnsupportedByLOBMatching_TypeError(self):
        buy_dict = {"ID":[7], "Price":[98], "Quantity":[10], "Time":[dt.datetime(2024,9,1,10,42,0)], "LMMProp":[0.1]}
        sell_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[], "LMMProp":[]}
        matching_rules = "LMM+FIFO"
        expected_exception = TypeError
        self.assertRaises(expected_exception, GetFullMatching, buy_dict, sell_dict, matching_rules)
    
    def test_TopLMMProRataAlgorithm_getLOBs_sameLOBs(self):
        buy_dict = {"ID":[2,3], 
                    "Price":[99,98], 
                    "Time":[dt.datetime(2024,9,1,10,30,1), dt.datetime(2024,9,1,10,30,4)],
                    "Quantity":[1,3]}
        sell_dict = {"ID":[], "Price":[], "Time":[], "Quantity":[]}
        expected_sell_dict = sell_dict
        expected_buy_dict = buy_dict
        matching_rules = "FIFO"

        FIFOMatchingAlgo = GetFullMatching(buy_dict, sell_dict, matching_rules)
        new_buy_dict, new_sell_dict = FIFOMatchingAlgo.getLOBDicts()

        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict, expected_buy_dict)

    
    def test_TopLMMProRataAlgorithm_noOpposingBuyOrderFIFO_addsToLOB(self):
        buy_dict = {"ID":[2,3], 
                    "Price":[99,98], 
                    "Time":[dt.datetime(2024,9,1,10,30,1), dt.datetime(2024,9,1,10,30,4)],
                    "Quantity":[1,3]}
        sell_dict = {"ID":[], "Price":[], "Time":[], "Quantity":[]}

        new_order = {"Side":"Buy","ID":5, "Price":100,"Time":dt.datetime(2024,9,1,10,30,10),"Quantity":3}
        
        expected_buy_dict = {"ID":[5,2,3], 
                             "Price":[100,99,98], 
                             "Time":[dt.datetime(2024,9,1,10,30,10),dt.datetime(2024,9,1,10,30,1), dt.datetime(2024,9,1,10,30,4)],
                             "Quantity":[3,1,3]}
        expected_sell_dict = {"ID":[], "Price":[], "Time":[], "Quantity":[]}
        
        matching_rules = "FIFO"

        FIFOMatchingAlgo = GetFullMatching(buy_dict, sell_dict, matching_rules)
        FIFOMatchingAlgo.processNewOrder(new_order)
        new_buy_dict, new_sell_dict = FIFOMatchingAlgo.getLOBDicts()
        
        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict, expected_buy_dict)
    
    def test_TopLMMProRataAlgorithm_noOpposingSellOrderFIFO_addsToDict(self):
        buy_dict = {"ID":[], "Price":[], "Time":[], "Quantity":[]}
        sell_dict = {"ID":[1,4], 
                     "Price":[100,100],
                      "Time":[dt.datetime(2024,9,1,10,30,3), dt.datetime(2024,9,1,10,30,10)],
                      "Quantity":[4,7]}

        new_order = {"Side":"Sell","ID":6, "Price":102, "Time":dt.datetime(2024,9,1,10,30,11),"Quantity":2}

        expected_buy_dict = {"ID":[], "Price":[], "Time":[], "Quantity":[]}
        expected_sell_dict = {"ID":[1,4,6],
                              "Price":[100,100,102], 
                              "Time":[dt.datetime(2024,9,1,10,30,3), dt.datetime(2024,9,1,10,30,10),dt.datetime(2024,9,1,10,30,11)],
                              "Quantity":[4,7,2]}
        
        matching_rules = "FIFO"

        FIFOMatchingAlgo = GetFullMatching(buy_dict, sell_dict, matching_rules)
        FIFOMatchingAlgo.processNewOrder(new_order)
        new_buy_dict, new_sell_dict = FIFOMatchingAlgo.getLOBDicts()

        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict, expected_buy_dict)
    
    def test_TopLMMProRataAlgorithm_matchableBuyOrderFIFO_matchesOneOrder(self):
        buy_dict = {"ID":[2,3],
                    "Price":[99,98], 
                    "Time":[dt.datetime(2024,9,1,10,30,1), dt.datetime(2024,9,1,10,30,4)],
                    "Quantity":[3,5]}
        sell_dict = {"ID":[1,4], 
                     "Price":[100,100],
                      "Time":[dt.datetime(2024,9,1,10,30,3), dt.datetime(2024,9,1,10,30,10)],
                      "Quantity":[4,7]}
        
        new_order = {"Side":"Buy","ID":7,"Price":100,"Time":dt.datetime(2024,9,1,10,30,11),"Quantity":6} 
        
        expected_buy_dict = {"ID":[2,3],
                    "Price":[99,98], 
                    "Time":[dt.datetime(2024,9,1,10,30,1), dt.datetime(2024,9,1,10,30,4)],
                    "Quantity":[3,5]}
        expected_sell_dict = {"ID":[4], 
                     "Price":[100],
                      "Time":[dt.datetime(2024,9,1,10,30,10)],
                      "Quantity":[5]}
        
        matching_rules = "FIFO"
        
        FIFOMatchingAlgo = GetFullMatching(buy_dict, sell_dict, matching_rules)
        FIFOMatchingAlgo.processNewOrder(new_order)
        new_buy_dict, new_sell_dict = FIFOMatchingAlgo.getLOBDicts()

        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict, expected_buy_dict)

    def test_TopLMMProRataAlgorithm_matchableSellOrderFIFO_matchesOneOrder(self):
        buy_dict = {"ID":[2,3],
                    "Price":[99,98], 
                    "Time":[dt.datetime(2024,9,1,10,30,1), dt.datetime(2024,9,1,10,30,4)],
                    "Quantity":[3,5]}
        sell_dict = {"ID":[1,4], 
                    "Price":[100,100],
                    "Time":[dt.datetime(2024,9,1,10,30,3), dt.datetime(2024,9,1,10,30,10)],
                    "Quantity":[4,7]}

        new_order = {"Side":"Sell",
                    "ID":8, 
                    "Price":95, 
                    "Time":dt.datetime(2024,9,1,10,30,14),
                    "Quantity":9}
        
        # 9 > 3+5 so it should wipe out both buy_dict orders and then become a sell order:
        expected_buy_dict = {"ID":[], "Price":[], "Time":[], "Quantity":[]}
        expected_sell_dict = {"ID":[8,1,4], "Price":[95,100,100], 
                            "Time":[dt.datetime(2024,9,1,10,30,14),dt.datetime(2024,9,1,10,30,3), dt.datetime(2024,9,1,10,30,10)],
                            "Quantity":[1,4,7]}
        matching_rules = "FIFO"

        FIFOMatchingAlgo = GetFullMatching(buy_dict, sell_dict, matching_rules)
        FIFOMatchingAlgo.processNewOrder(new_order)
        new_buy_dict, new_sell_dict = FIFOMatchingAlgo.getLOBDicts()

        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict, expected_buy_dict)
    
    def test_TopLMMProRataAlgorithm_matchableBuyOrderProRata_matchesProportionally(self):
        buy_dict = {"ID":[7], "Price":[98], "Quantity":[10], "Time":[dt.datetime(2024,9,1,10,42,0)]}
        sell_dict = {"ID":[5,2,3],
                     "Price":[99,100,100],
                     "Quantity":[20,30,60],
                     "Time":[dt.datetime(2024,9,1,10,40,0),dt.datetime(2024,9,1,10,36,0), dt.datetime(2024,9,1,10,37,0)]}
        new_order = {"Side":"Buy", "ID":8, "Price":100, "Quantity":56, "Time":dt.datetime(2024,9,1,10,43,0)}
        # Should match 20 with the $99 orders, leaving 36.
        # 36 / (30+60) * 30 = 12
        # 36 / (30+60) * 60 = 24
        # So after matching with the $100 orders, we get:
        # 30 - 12 = 18 shares in order 3
        # 60 - 24 = 36 shares in order 5.
        # (And no change to the buy dictionary)
        expected_buy_dict = {"ID":[7], "Price":[98], "Quantity":[10], "Time":[dt.datetime(2024,9,1,10,42,0)]}
        expected_sell_dict = {"ID":[2,3],
                     "Price":[100,100],
                     "Quantity":[18,36],
                     "Time":[dt.datetime(2024,9,1,10,36,0), dt.datetime(2024,9,1,10,37,0)]}
        matching_rules = "ProRata"
        ProRataMatchingAlgo = GetFullMatching(buy_dict, sell_dict, matching_rules)
        ProRataMatchingAlgo.processNewOrder(new_order)
        new_buy_dict, new_sell_dict = ProRataMatchingAlgo.getLOBDicts()

        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict, expected_buy_dict)
    
    def test_TopLMMProRataAlgorithm_partialMatchableBuyOrderProRata_matchThenAdd(self):
        # Testing that a buy order is matched until it fulfills all
        # sell orders until it's added to the LOB.
        buy_dict = {"ID":[7], "Price":[98], "Quantity":[10], "Time":[dt.datetime(2024,9,1,10,42,0)]}
        sell_dict = {"ID":[5,2,3],
                     "Price":[99,100,100],
                     "Quantity":[20,30,60],
                     "Time":[dt.datetime(2024,9,1,10,40,0),dt.datetime(2024,9,1,10,36,0), dt.datetime(2024,9,1,10,37,0)]}
        
        new_order = {"Side":"Buy", "ID":8, "Price":99, "Quantity":56, "Time":dt.datetime(2024,9,1,10,43,0)}

        # Price is 99 so it matches with order 5 until order 5 is exhausted,
        # then is added to the buy dict.

        expected_buy_dict = {"ID":[8,7], "Price":[99,98], "Quantity":[36,10], 
                             "Time":[dt.datetime(2024,9,1,10,43,0),dt.datetime(2024,9,1,10,42,0)]}
        expected_sell_dict = {"ID":[2,3],"Price":[100,100],"Quantity":[30,60],
                              "Time":[dt.datetime(2024,9,1,10,36,0), dt.datetime(2024,9,1,10,37,0)]}
        matching_rules = "ProRata"
        ProRataMatchingAlgo = GetFullMatching(buy_dict, sell_dict, matching_rules)
        ProRataMatchingAlgo.processNewOrder(new_order)
        new_buy_dict, new_sell_dict = ProRataMatchingAlgo.getLOBDicts()

        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict, expected_buy_dict)
    
    def test_TopLMMProRataAlgorithm_partialMatchableSellOrderProRata_match(self):
        buy_dict = {"ID":[11,7,9,14], 
            "Price":[99,98,98,98], 
            "Quantity":[10,30,40,80], 
            "Time":[dt.datetime(2024,9,1,10,44,0),dt.datetime(2024,9,1,10,42,0),dt.datetime(2024,9,1,10,44,0),dt.datetime(2024,9,1,10,47,0)]
        }
        sell_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}
        
        new_order = {"Side":"Sell", "ID":15, "Price":98, "Quantity":40, "Time":dt.datetime(2024,9,1,10,49,0)}

        expected_buy_dict = {"ID":[7,9,14], 
            "Price":[98,98,98], 
            "Quantity":[24,32,64], 
            "Time":[dt.datetime(2024,9,1,10,42,0),dt.datetime(2024,9,1,10,44,0),dt.datetime(2024,9,1,10,47,0)]
        }
        expected_sell_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}

        matching_rules = "ProRata"
        ProRataMatchingAlgo = GetFullMatching(buy_dict, sell_dict, matching_rules)
        ProRataMatchingAlgo.processNewOrder(new_order)
        new_buy_dict, new_sell_dict = ProRataMatchingAlgo.getLOBDicts()

        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict, expected_buy_dict)
    
    def test_TopLMMProRataAlgorithm_partialMatchableSellOrderProRata_matchWithFIFO(self):
        buy_dict = {"ID":[7,9,14], 
            "Price":[98,98,98], 
            "Quantity":[24,32,64], 
            "Time":[dt.datetime(2024,9,1,10,42,0),dt.datetime(2024,9,1,10,44,0),dt.datetime(2024,9,1,10,47,0)]
        }
        sell_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}
        new_order = {"Side":"Sell", "ID":16, "Price":97, "Quantity":22, "Time":dt.datetime(2024,9,1,10,43,0)}

        expected_buy_dict = {"ID":[7,9,14], 
            "Price":[98,98,98], 
            "Quantity":[18,27,53], 
            "Time":[dt.datetime(2024,9,1,10,42,0),dt.datetime(2024,9,1,10,44,0),dt.datetime(2024,9,1,10,47,0)]
        }
        expected_sell_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}

        matching_rules = "ProRata"
        ProRataMatchingAlgo = GetFullMatching(buy_dict, sell_dict, matching_rules)
        ProRataMatchingAlgo.processNewOrder(new_order)
        new_buy_dict, new_sell_dict = ProRataMatchingAlgo.getLOBDicts()

        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict, expected_buy_dict)


        # —— Case (1)：只有 LMM（无 Top），卖出 1750@100 —— 
    def test_LMM_only_case1_sell1750_at100(self):
        buy_dict = {"ID":[1,3,4,2,5],
                    "Price":[100,100,100,99,98],
                    "Quantity":[50,1512,2300,500,1000],
                    "Time":[dt.datetime(2024,9,1,10,40,0),
                            dt.datetime(2024,9,1,10,42,0),
                            dt.datetime(2024,9,1,10,43,0),
                            dt.datetime(2024,9,1,10,41,0),
                            dt.datetime(2024,9,1,10,44,0)],
                    "Top":[0,0,0,0,0],
                    "LMMProp":[0.1,0.05,0.1,0.05,0.0],
                    "LMMID":["LMM3","LMM2","LMM3","LMM2","N"]}
        sell_dict = {"ID":[], "Price":[], "Quantity":[], "Time":[],
                     "Top":[], "LMMProp":[], "LMMID":[]}
        new_order = {"Side":"Sell","ID":99,"Price":100,"Quantity":1750,
                     "Time":dt.datetime(2024,9,1,10,50,0)}
        algo = GetFullMatching(buy_dict, sell_dict, "LMM+ProRata")
        algo.processNewOrder(new_order)
        out_buy, out_sell = algo.getLOBDicts()
        # 预期：LMM3 拿 175（50 给 ID=1，125 给 ID=4），LMM2 拿 87（全给 ID=3），
        # 剩余 1488 做 Pro Rata -> ID=3:1425-589=836，ID=4:2175-899=1276
        expected_buy = {"ID":[3,4,2,5],
                        "Price":[100,100,99,98],
                        "Quantity":[836,1276,500,1000],
                        "Time":[dt.datetime(2024,9,1,10,42,0),
                                dt.datetime(2024,9,1,10,43,0),
                                dt.datetime(2024,9,1,10,41,0),
                                dt.datetime(2024,9,1,10,44,0)],
                        "Top":[0,0,0,0],
                        "LMMProp":[0.05,0.1,0.05,0.0],
                        "LMMID":["LMM2","LMM3","LMM2","N"]}
        self.assertDictEqual(out_sell, sell_dict)   # 卖簿应保持空
        self.assertDictEqual(out_buy, expected_buy)

    # —— Case (2)：仅 Top + FIFO；卖出 1500@100 —— 
    def test_Top_plus_FIFO_case2_sell1500_at100(self):
        buy_dict = {"ID":[1,3,4,2,5],
                    "Price":[100,100,100,99,98],
                    "Quantity":[300,1512,2300,500,1000],
                    "Time":[dt.datetime(2024,9,1,10,40,0),
                            dt.datetime(2024,9,1,10,42,0),
                            dt.datetime(2024,9,1,10,43,0),
                            dt.datetime(2024,9,1,10,41,0),
                            dt.datetime(2024,9,1,10,44,0)],
                    "Top":[250,0,0,0,0]}
        sell_dict = {"ID":[], "Price":[], "Quantity":[], "Time":[], "Top":[]}
        new_order = {"Side":"Sell","ID":77,"Price":100,"Quantity":1500,
                     "Time":dt.datetime(2024,9,1,10,50,0)}
        algo = GetFullMatching(buy_dict, sell_dict, "Top+FIFO")
        algo.processNewOrder(new_order)
        out_buy, out_sell = algo.getLOBDicts()
        # 逻辑：先吃 Top 250（给 ID=1，余 50@ID=1）；剩 1250 用 FIFO：
        # 先吃 ID=1 的 50，余 1200；再吃 ID=3 的 1200 -> ID=3 余 312。
        expected_buy = {"ID":[3,4,2,5],
                        "Price":[100,100,99,98],
                        "Quantity":[312,2300,500,1000],
                        "Time":[dt.datetime(2024,9,1,10,42,0),
                                dt.datetime(2024,9,1,10,43,0),
                                dt.datetime(2024,9,1,10,41,0),
                                dt.datetime(2024,9,1,10,44,0)],
                        "Top":[0,0,0,0]}
        self.assertDictEqual(out_sell, sell_dict)
        self.assertDictEqual(out_buy, expected_buy)

    # —— Case (3)：Top + LMM + ProRata；卖出 2000@100 —— 
    def test_Top_LMM_ProRata_case3_sell2000_at100(self):
        buy_dict = {"ID":[1,3,4,2,5],
                    "Price":[100,100,100,99,98],
                    "Quantity":[300,1512,2300,500,1000],
                    "Time":[dt.datetime(2024,9,1,10,40,0),
                            dt.datetime(2024,9,1,10,42,0),
                            dt.datetime(2024,9,1,10,43,0),
                            dt.datetime(2024,9,1,10,41,0),
                            dt.datetime(2024,9,1,10,44,0)],
                    "Top":[250,0,0,0,0],
                    "LMMProp":[0.1,0.05,0.1,0.05,0.0],
                    "LMMID":["LMM3","LMM2","LMM3","LMM2","N"]}
        sell_dict = {"ID":[], "Price":[], "Quantity":[], "Time":[],
                     "Top":[], "LMMProp":[], "LMMID":[]}
        new_order = {"Side":"Sell","ID":88,"Price":100,"Quantity":2000,
                     "Time":dt.datetime(2024,9,1,10,50,0)}
        algo = GetFullMatching(buy_dict, sell_dict, "Top+LMM+ProRata")
        algo.processNewOrder(new_order)
        out_buy, out_sell = algo.getLOBDicts()
        # 与讲解一致：Top 先吃 250（剩 1750），LMM3 取 175（50 到 ID=1，125 到 ID=4），
        # LMM2 取 87（到 ID=3），剩 1488 做 Pro Rata -> 836/1276
        expected_buy = {"ID":[3,4,2,5],
                        "Price":[100,100,99,98],
                        "Quantity":[836,1276,500,1000],
                        "Time":[dt.datetime(2024,9,1,10,42,0),
                                dt.datetime(2024,9,1,10,43,0),
                                dt.datetime(2024,9,1,10,41,0),
                                dt.datetime(2024,9,1,10,44,0)],
                        "Top":[0,0,0,0],
                        "LMMProp":[0.05,0.1,0.05,0.0],
                        "LMMID":["LMM2","LMM3","LMM2","N"]}
        self.assertDictEqual(out_sell, sell_dict)
        self.assertDictEqual(out_buy, expected_buy)

    # —— Top 分配：更优价但数量 < 50（不授予 top）——
    def test_Top_assign_better_price_below_threshold(self):
        buy_dict = {"ID":[10], "Price":[100], "Quantity":[200],
                    "Time":[dt.datetime(2024,9,1,10,40,0)], "Top":[0]}
        sell_dict = {"ID":[], "Price":[], "Quantity":[], "Time":[], "Top":[]}
        new_order = {"Side":"Buy","ID":11,"Price":101,"Quantity":30,
                     "Time":dt.datetime(2024,9,1,10,41,0)}
        algo = GetFullMatching(buy_dict, sell_dict, "Top+FIFO")
        algo.processNewOrder(new_order)
        out_buy, _ = algo.getLOBDicts()
        # 新单量 < 50，不给 top
        self.assertEqual(out_buy["Top"][0], 0)      # 新来的在最前，Top 应为 0

    # —— Top 分配：更优价且 50 ≤ 量 < 250 —— 
    def test_Top_assign_better_price_between_threshold_and_cap(self):
        buy_dict = {"ID":[10], "Price":[100], "Quantity":[200],
                    "Time":[dt.datetime(2024,9,1,10,40,0)], "Top":[0]}
        sell_dict = {"ID":[], "Price":[], "Quantity":[], "Time":[], "Top":[]}
        new_order = {"Side":"Buy","ID":11,"Price":101,"Quantity":120,
                     "Time":dt.datetime(2024,9,1,10,41,0)}
        algo = GetFullMatching(buy_dict, sell_dict, "Top+FIFO")
        algo.processNewOrder(new_order)
        out_buy, _ = algo.getLOBDicts()
        self.assertEqual(out_buy["Top"][0], 120)    # 直接给足 120（< 250）

    # —— Top 分配：更优价且数量 > 250（封顶为 250）—— 
    def test_Top_assign_better_price_above_cap(self):
        buy_dict = {"ID":[10], "Price":[100], "Quantity":[200],
                    "Time":[dt.datetime(2024,9,1,10,40,0)], "Top":[0]}
        sell_dict = {"ID":[], "Price":[], "Quantity":[], "Time":[], "Top":[]}
        new_order = {"Side":"Buy","ID":11,"Price":101,"Quantity":400,
                     "Time":dt.datetime(2024,9,1,10,41,0)}
        algo = GetFullMatching(buy_dict, sell_dict, "Top+FIFO")
        algo.processNewOrder(new_order)
        out_buy, _ = algo.getLOBDicts()
        self.assertEqual(out_buy["Top"][0], 250)    # 封顶 250

    # —— Top 分配：与市场持平（不授予 top）——
    def test_Top_assign_tie_price(self):
        buy_dict = {"ID":[10], "Price":[100], "Quantity":[200],
                    "Time":[dt.datetime(2024,9,1,10,40,0)], "Top":[0]}
        sell_dict = {"ID":[], "Price":[], "Quantity":[], "Time":[], "Top":[]}
        new_order = {"Side":"Buy","ID":11,"Price":100,"Quantity":120,
                     "Time":dt.datetime(2024,9,1,10,41,0)}
        algo = GetFullMatching(buy_dict, sell_dict, "Top+FIFO")
        algo.processNewOrder(new_order)
        out_buy, _ = algo.getLOBDicts()
        self.assertEqual(out_buy["Top"][0], 0)      # 与最优价相同 -> 不是 top

    # —— Top 分配：同侧为空时（成立新最优价）——
    def test_Top_assign_when_same_side_empty(self):
        buy_dict = {"ID":[], "Price":[], "Quantity":[], "Time":[], "Top":[]}
        sell_dict = {"ID":[], "Price":[], "Quantity":[], "Time":[], "Top":[]}
        new_order = {"Side":"Buy","ID":11,"Price":101,"Quantity":120,
                     "Time":dt.datetime(2024,9,1,10,41,0)}
        algo = GetFullMatching(buy_dict, sell_dict, "Top+FIFO")
        algo.processNewOrder(new_order)
        out_buy, _ = algo.getLOBDicts()
        self.assertEqual(out_buy["Top"][0], 120)    # 空簿视作“改善市场”，授予 top

    # —— Top 分配：已有 100 的 top，新来 101 更优价 -> 旧 top 清零 —— 
    def test_Top_assign_new_better_price_resets_old_top(self):
        buy_dict = {"ID":[10], "Price":[100], "Quantity":[300],
                    "Time":[dt.datetime(2024,9,1,10,40,0)], "Top":[200]}
        sell_dict = {"ID":[], "Price":[], "Quantity":[], "Time":[], "Top":[]}
        new_order = {"Side":"Buy","ID":11,"Price":101,"Quantity":120,
                     "Time":dt.datetime(2024,9,1,10,41,0)}
        algo = GetFullMatching(buy_dict, sell_dict, "Top+FIFO")
        algo.processNewOrder(new_order)
        out_buy, _ = algo.getLOBDicts()
        # 新单在最前，旧单在第二行；旧单 top 应被清成 0
        self.assertEqual(out_buy["Top"][1], 0)
        self.assertEqual(out_buy["Top"][0], 120)


def TopLMMProRataMatchingClassSuite():

    loader = unittest.TestLoader()
    full_matching_suite = loader.loadTestsFromTestCase(TestTopLMMProRataClass)
    return full_matching_suite

runner.run(TopLMMProRataMatchingClassSuite())

# %%
class TopLMMProRataAlgorithm(BaseMatchingAlgorithm):
    def __init__(self, LOB_buy, LOB_sell, matching_rules):
        
        matching_list = set(matching_rules.split("+"))
        supported_list = set(["Top","LMM","ProRata", "FIFO"])
        required_list = set(["ProRata", "FIFO"])
        # LOB_list = set(["Top","LMM"])
        if not matching_list <= supported_list:
            raise ValueError("Matching type not supported. Entries must be one of: 'Top', 'LMM', 'ProRata', 'FIFO'")
        if len(matching_list & required_list) !=1:
            raise ValueError("Must have exactly one of ProRata, FIFO matching specified")
        self.do_matching = {"Top": False, "LMM":False,"ProRata":False, "FIFO":False}
    
        for matching_type in matching_list:
            if matching_type == "LMM":
                LMMProp_available = ( "LMMProp" in LOB_buy.orders and "LMMProp" in LOB_sell.orders)
                LMMID_available = ( "LMMID" in LOB_buy.orders and "LMMID" in LOB_sell.orders)
                if LMMProp_available and LMMID_available:
                    self.do_matching[matching_type] = True
                else:
                    raise TypeError(f"{matching_type} not supported by LOB inputted.")
            elif matching_type == "Top":
                if "Top" in LOB_buy.orders and "Top" in LOB_sell.orders:
                    self.do_matching[matching_type] = True
                else:
                    raise TypeError(f"{matching_type} not supported by LOB inputted.")
            else:
                self.do_matching[matching_type] = True
        super().__init__(LOB_buy, LOB_sell)
        self.min_top_order = 50
        self.max_top_amount = 250
    
    def insertOrderIntoLOB(self, new_order):
        # Checking if the new order needs a Top amount:
        if self.do_matching["Top"]:
            if new_order["Quantity"] >= self.min_top_order:
                top_amt = min(new_order["Quantity"], self.max_top_amount)
            else:
                top_amt = 0
            # Check if it beats the market:
            temp_order = new_order.copy()
            # Swap sides, then check if it's matchable. 
            temp_order["Side"] = new_order["OpposingSide"]
            temp_order["OpposingSide"] = new_order["Side"]
            # If the incoming order price is matchable, then the best price is at least
            # as good as the order price. Therefore, the new order doesn't beat the market
            # Otherwise, we should add a top amount. 
            matchable = self.checkIfMatch(temp_order)
            if not matchable:
                new_order["Top"] = top_amt
            else:
                new_order["Top"] = 0
            top_orders = self.LOB_dict[new_order["Side"]].orders["Top"] 
            new_top_orders = [0 for i in top_orders]
            self.LOB_dict[new_order["Side"]].orders["Top"]  = new_top_orders
        super().insertOrderIntoLOB(new_order)

    def topMatching(self, new_order):
        # Performs Top order matching of the incoming order
        # May assume we've already checked that it's matchable.
        best_order = self.LOB_dict[new_order["OpposingSide"]].getOrder(0)
        top_amount = best_order["Top"]
        total_amount = best_order["Quantity"] 
        quantity = new_order["Quantity"]
        finished = False
        if top_amount > 0:
            if quantity >= top_amount:
                # If we trade off the top amount, we remove this from the new order quantity, and 
                # then remove the top amount from the current best order.
                quantity = quantity - top_amount
                new_order["Quantity"] = quantity
                if total_amount == top_amount:
                    # Delete the best order when appropriate
                    self.deleteOpposingOrder(new_order)
                else:
                    self.LOB_dict[new_order["OpposingSide"]].modifyOrder(0, "Top", 0)

                    self.LOB_dict[new_order["OpposingSide"]].modifyOrder(0, "Quantity", total_amount - top_amount)
                # Return whether the loop has finished.
                
                matchable = self.checkIfMatch(new_order)
                if quantity == 0:
                    finished = True
                elif not matchable:
                    self.insertOrderIntoLOB(new_order)
                    finished = True
            else:
                # Trade all of the incoming order.
                quantity = 0
                new_order["Quantity"] = quantity
                self.LOB_dict[new_order["OpposingSide"]].modifyOrder(0, "Top", top_amount - quantity)
                self.LOB_dict[new_order["OpposingSide"]].modifyOrder(0, "Quantity", total_amount - quantity)
                finished = True
        return new_order, finished

    def proRataOneStep(self, new_order):
        # We may need to check if LMM exist at the next best price, so we create a single step ProRata 
        # that runs for just the current best price and then stops. It returns whether the order has
        # terminated, so this can be used by other methods if necessary.
        best_order_quantities = self.LOB_dict[new_order["OpposingSide"]].getBestOrderQuantities()
        total_quantity = sum(best_order_quantities)
        quantity = new_order["Quantity"]
        finished = False
        if quantity >= total_quantity:
            # Delete all of the orders with the best price
            for i in range(len(best_order_quantities)):
                self.deleteOpposingOrder(new_order)
            quantity = quantity - total_quantity
            new_order["Quantity"] = quantity
            matchable = self.checkIfMatch(new_order)
            if quantity == 0:
                finished =True
            elif not matchable:
                self.insertOrderIntoLOB(new_order)
                finished = True
        else:
            traded_quantities = np.floor(np.array(best_order_quantities)/total_quantity * quantity)
            remaining_quantities = best_order_quantities - traded_quantities
            quantity = quantity - np.sum(traded_quantities)
            self.LOB_dict[new_order["OpposingSide"]].orders["Quantity"][0:len(best_order_quantities)] = remaining_quantities.tolist()
            new_order["Quantity"] = quantity
            if quantity != 0:
                self.priceTimeMatchOrder(new_order)
            finished = True
        return new_order, finished

    def LMMOneStep(self, new_order):
        # Performs the LMM priority matching at the best price. 
        # This needs to handle FIFO matching when an LMM has multiple orders.
        best_order_quantities = self.LOB_dict[new_order["OpposingSide"]].getBestOrderQuantities()
        best_orders = [self.LOB_dict[new_order["OpposingSide"]].getOrder(i) for i in range(len(best_order_quantities))]
        best_order_LMMProps = [order["LMMProp"] for order in best_orders]
        best_price = self.LOB_dict[new_order["OpposingSide"]].orders["Price"][0]
        
        if max(best_order_LMMProps) == 0:
            # If there aren't any LMM orders, we return the order
            # and that we haven't finished matching.
            return new_order, False, False
        
        # Otherwise, we start LMM matching.
        best_order_LMMIDs = [order["LMMID"] for order in best_orders]
        LMM_prop_dict = {ID:Prop for ID, Prop in zip(best_order_LMMIDs,best_order_LMMProps)}
        finished = False
        if "N" in LMM_prop_dict:
            LMM_prop_dict.pop("N")
        total_quantity = 0
        quantity = new_order["Quantity"]
        for key in LMM_prop_dict:
            
            max_LMM_quantity = np.floor(LMM_prop_dict[key]*quantity)
            untraded_quantity = max_LMM_quantity
            # Implement FIFO matching on with this quantity on only orders with the ID specified by key:
            finished_current_key = False
            next_idx = best_order_LMMIDs.index(key)
            while untraded_quantity > 0 and not finished_current_key:
                next_quantity= best_orders[next_idx]["Quantity"]
                if untraded_quantity >= next_quantity:
                    # Complete and remove this order, move on to the next:
                    untraded_quantity = untraded_quantity - next_quantity
                    self.deleteOpposingOrder(new_order, LOB_index = next_idx)
                    # Delete orders to match indexing
                    best_order_LMMIDs.pop(next_idx)
                    best_orders.pop(next_idx)
                else:
                    next_quantity = next_quantity - untraded_quantity
                    untraded_quantity = 0
                    self.LOB_dict[new_order["OpposingSide"]].modifyOrder(next_idx, "Quantity", next_quantity)
                    best_order_LMMIDs[next_idx] = "Done"
                    
                if key in best_order_LMMIDs:
                    next_idx = best_order_LMMIDs.index(key)
                else:
                    finished_current_key = True
            # Only count the quantity that was traded.
            total_quantity += max_LMM_quantity - untraded_quantity
        
        quantity = quantity - total_quantity
        new_order["Quantity"] = quantity
        # Reports whether there are any more (matchable) orders to trade, 
        # and whether it finished all LMM orders
        matchable = self.checkIfMatch(new_order)

        if not matchable:
            self.insertOrderIntoLOB(new_order)
            finished = True
        # If the price has moved, flag that we should do LMM again to the main algorithm
        if best_price == self.LOB_dict[new_order["OpposingSide"]].orders["Price"][0]:
            do_LMM = False
        else:
            do_LMM = True
        return new_order, finished, do_LMM

    def priceTimeCurrentPrice(self, new_order):
        # A method to perform price time matching until we use up all orders at
        # the current price (or the incoming order)
        best_order_price = self.LOB_dict[new_order["OpposingSide"]].orders["Price"][0]
        current_order_price = best_order_price
        finished = False
        # Loop until either the new order is matched or we run out of orders
        # in the LOB at the best price, when we may need to use LMM matching 
        # again and should stop.
        while (not finished) and (current_order_price == best_order_price):
            # Perform one step of price time (FIFO) matching
            new_order, finished = self.priceTimeOneStep(new_order)
            if finished:
                return new_order, finished
            # Set the current order price to the price of the newest order
            current_order_price = self.LOB_dict[new_order["OpposingSide"]].orders["Price"][0]

        return new_order, finished

    def matchNewOrder(self, new_order):
        quantity = new_order["Quantity"]
        matchable = self.checkIfMatch(new_order)
        # If it's not matchable we stop
        if not matchable:
            self.insertOrderIntoLOB(new_order)
            return
        # Otherwise, we can match it so it's matchable with the current top order (if we're using top orders)
        if self.do_matching["Top"]:
            new_order, finished = self.topMatching(new_order)
            if finished:
                return
        # We set a boolean flag for whether we should do LMM matching. This only changes when we match
        # enough orders for the best price to go down.
        if self.do_matching["LMM"]:
            LMM_this_loop = True
        else:
            LMM_this_loop = False
        while quantity > 0:
            # LMM matching occurs on each loop
            if self.do_matching["LMM"] and LMM_this_loop:
                # Perform LMM matching and check if the incoming order was fulfilled. 
                # If so, we stop. Also check if we're doing LMM matching again next
                # (Which occurs whenever LMM matching uses all orders with the current best price.)
                new_order, finished, LMM_this_loop = self.LMMOneStep(new_order)
                if finished:
                    return
            if self.do_matching["ProRata"] and not LMM_this_loop:
                # Perform pro rata matching and check if the incoming order was fulfilled. 
                # If so, we stop.
                new_order, finished = self.proRataOneStep(new_order)
                if finished:
                    return
                # After pro rata matching, we'll always either finish or
                # be on a new price, so the LMM flag is reset.
                if self.do_matching["LMM"]:
                    LMM_this_loop = True
            if self.do_matching["FIFO"] and not LMM_this_loop:
                new_order, finished = self.priceTimeCurrentPrice(new_order)
                if finished:
                    return
                if self.do_matching["LMM"]:
                    LMM_this_loop = True
            quantity = new_order["Quantity"]

runner.run(TopLMMProRataMatchingClassSuite())

# %% [markdown]
# # 3.4 - Other order types - Orders with only immediate effects

# %%
class SeveralOrderMatchingAlgorithm(TopLMMProRataAlgorithm):
    def __init__(self, LOB_buy, LOB_sell, matching_rules, min_price_var = 1):
        super().__init__(LOB_buy, LOB_sell, matching_rules)
        self.minimum_price_variation = min_price_var

    def processNewOrder(self, new_order):
        # We'll call this method to add a new order to the LOB / match it.

        # 1) 补上对手盘方向（"OpposingSide"），便于访问另一侧 LOB
        new_order = self.addOpposingSide(new_order)

        # 2) 若未显式指定类型/时效，设默认："Limit" + "Standard"
        new_order = self.addOrderType(new_order)

        # 3) 对有“即时生效特性”的订单（Market / IOC / ALO）先做预处理
        #    若需要直接取消或只做属性修改，这里返回 finished=True/False
        new_order, finished = self.checkNewOrder(new_order)
        if finished:
            return
        
        # 4) 其余情况走父类的撮合流程（其中会处理 Top/LMM/ProRata/FIFO）
        self.matchNewOrder(new_order)

    def addOrderType(self, new_order):
        # 若没有给出 "Type"（订单类型），默认当作“限价单”
        if "Type" not in new_order:
            new_order["Type"] = "Limit"
        # 若没有给出 "TimeType"（时效/附加条件），默认“标准单”
        if "TimeType" not in new_order:
            new_order["TimeType"] = "Standard"
        return new_order
    
    def checkIfMatch(self, new_order):
        # 市价单：只要对手盘**还有挂单**，就认为“可撮合”
        if new_order["Type"] == "Market":
            opp = new_order["OpposingSide"]
            return len(self.LOB_dict[opp].orders["ID"]) > 0

        # 其它类型保持父类逻辑（跨价差才可撮合）
        return super().checkIfMatch(new_order)

    def checkNewOrder(self, new_order):
        finished = False
        order_type = new_order.get("Type", "Limit")
        time_type  = new_order.get("TimeType", "Standard")
        side       = new_order["Side"]
        opp        = new_order["OpposingSide"]
        opp_orders = self.LOB_dict[opp].orders
        has_opposing = len(opp_orders["ID"]) > 0

        # 1) 市价单对手盘为空 -> 取消
        if order_type == "Market" and not has_opposing:
            return new_order, True

        # 2) IOC：把数量截断到“最优价同层的总量”；限价 IOC 不可撮合则取消
        if time_type == "IOC":
            best_q = self.LOB_dict[opp].getBestOrderQuantities() if has_opposing else None
            total_at_best = sum(best_q) if best_q else 0
            if order_type == "Limit" and not super().checkIfMatch(new_order):
                return new_order, True
            new_order["Quantity"] = min(new_order["Quantity"], total_at_best)
            if new_order["Quantity"] == 0:
                return new_order, True

        # 3) ALO：无论写在 Type 还是 TimeType，都要处理
        if (order_type == "ALO" or time_type == "ALO") and has_opposing:
            # 只有“会撮合”的价格才需要改价
            if super().checkIfMatch(new_order):
                best_price = opp_orders["Price"][0]
                tick = self.minimum_price_variation
                if side == "Buy":
                    # 买单：压到卖一减一个最小跳动
                    new_order["Price"] = best_price - tick
                else:
                    # 卖单：抬到买一加一个最小跳动
                    new_order["Price"] = best_price + tick
            # 后续就把它当普通限价单用
            new_order["Type"] = "Limit"

        # 其余情况继续走撮合
        return new_order, finished
    
    def insertOrderIntoLOB(self, new_order):
        # 市价单永远**不**入簿（撮合不到就算“取消”）
        if new_order["Type"] == "Market":
            return
        # 其它类型沿用父类逻辑（会自动按侧别/时间排序插入）
        super().insertOrderIntoLOB(new_order)


# %% [markdown]
# ### 3.5 - New order types - Orders with delayed effects

# %%
class AllOrderMatchingAlgorithm(SeveralOrderMatchingAlgorithm):
    def __init__(self, LOB_buy, LOB_sell, matching_rules, minimum_price_var = 1,
                 buy_iceberg_dict = None, sell_iceberg_dict = None, buy_stop_dict = None, sell_stop_dict = None):
        super().__init__(LOB_buy, LOB_sell, matching_rules, 
                         min_price_var = minimum_price_var)
        if self.do_matching["LMM"]:
            default_iceberg = {"ID":[],"Price":[], "Quantity":[], "DisplayQuantity":[],
                               "Time":[], "LMMProp":[], "LMMID":[]}
            default_stop = {"ID":[],"Price":[], "Quantity":[], "DisplayQuantity":[], 
                            "Time":[],"LMMProp":[], "LMMID":[]}
        else:
            default_iceberg = {"ID":[],"Price":[], "Quantity":[], "DisplayQuantity":[], "Time":[]}
            default_stop = {"ID":[], "Price":[], "Quantity":[], "Time":[]}

        # —— 初始化四个辅助 LOB（买/卖 × 冰山/止损），注意要“深复制式”清空内部列表 ——
        if buy_iceberg_dict is None:
            buy_iceberg_dict = default_iceberg.copy()
            for key in buy_iceberg_dict:
                buy_iceberg_dict[key] = []
        if sell_iceberg_dict is None:
            sell_iceberg_dict = default_iceberg.copy()
            for key in sell_iceberg_dict:
                sell_iceberg_dict[key] = []
        if buy_stop_dict is None:
            buy_stop_dict = default_stop.copy()
            for key in buy_stop_dict:
                buy_stop_dict[key] = []
        if sell_stop_dict is None:
            sell_stop_dict = default_stop.copy()
            for key in sell_stop_dict:
                sell_stop_dict[key] = []
        # 止损的排序方向与正常 LOB 相反：买止损按“低→高”、卖止损按“高→低”
        stop_dict = {"Buy":LOBSide("Sell", buy_stop_dict), "Sell":LOBSide("Buy", sell_stop_dict)}
        iceberg_dict = {"Buy":LOBSide("Buy", buy_iceberg_dict), "Sell":LOBSide("Sell", sell_iceberg_dict)}
        self.stop_dict = stop_dict
        self.iceberg_dict = iceberg_dict
    
    def checkForIcebergs(self, side, order_id):
        # 若该 ID 不在相应侧的冰山列表里，直接返回“不命中”
        if order_id not in self.iceberg_dict[side].orders["ID"]:
            return None, False

        # 取出该冰山订单在冰山表中的位置
        iceberg_index = self.iceberg_dict[side].orders["ID"].index(order_id)
        # 显示量（每次放入 LOB 的单位量）
        display_quantity = self.iceberg_dict[side].orders["DisplayQuantity"][iceberg_index]
        # 剩余总量（尚未“显示/放入”的剩余总量）
        total_quantity = self.iceberg_dict[side].orders["Quantity"][iceberg_index]
        # 下次要放入 LOB 的价格
        order_price = self.iceberg_dict[side].orders["Price"][iceberg_index]
        # 实际要显示的一份是“min(显示量, 剩余总量)”
        order_quantity = min(display_quantity, total_quantity)
        # 用当前时间作为“放入时间”（按分钟精度）
        current_time = dt.datetime.now().replace(second=0, microsecond=0)

        # 若开启 LMM，需要把 LMMProp/LMMID 也随新的一份传递下去
        if self.do_matching["LMM"]:
            order_LMMProp = self.iceberg_dict[side].orders["LMMProp"][iceberg_index]
            order_LMMID = self.iceberg_dict[side].orders["LMMID"][iceberg_index]
            iceberg_limit_order = {"ID":order_id, "Side": side,"Type": "Limit",
                            "Price":order_price, "Quantity":order_quantity, "Time":current_time, 
                            "LMMProp":order_LMMProp, "LMMID":order_LMMID}
        else:
            iceberg_limit_order = {"ID":order_id, "Side": side, "Type": "Limit",
                            "Price":order_price, "Quantity":order_quantity, "Time":current_time}

        # —— 从冰山“总量”里扣掉这次要显示的一份；若用尽则把冰山记录整条删掉 ——
        if total_quantity > order_quantity:
            # 还有余量：把剩余总量改为“扣除后的值”
            new_total = total_quantity - order_quantity
            self.iceberg_dict[side].modifyOrder(iceberg_index, "Quantity", new_total)
        else:
            # 已无剩余：把冰山记录整条移除（该 ID 的冰山生命周期结束）
            self.iceberg_dict[side].deleteOrder(iceberg_index)

        # 返回“下一份显示单”以及“命中冰山”的标记
        return iceberg_limit_order, True

    def deleteOpposingOrder(self, new_order, LOB_index=0):
        opposing_side = new_order["OpposingSide"]                      # 对手盘侧别
        order_id = self.LOB_dict[opposing_side].orders["ID"][LOB_index]# 即将被删除的对手盘订单 ID

        # 先检查这个被删除的订单是不是冰山的一份（若是，要补下一份）
        iceberg_order, hit_iceberg = self.checkForIcebergs(opposing_side, order_id)

        # 调用父类逻辑真正把对手盘的该行删掉（完成当前这“一份”的撮合）
        super().deleteOpposingOrder(new_order, LOB_index = LOB_index)

        if hit_iceberg:
            # 若确实是冰山的一份：把“下一份显示量”立刻插回对手盘 LOB
            # 注意这里直接“入簿”（insert），不要走 processNewOrder，以避免递归重新走 Top/LMM 等步骤
            self.insertOrderIntoLOB(iceberg_order)

    def insertOrderIntoLOB(self, new_order):
        if new_order["Type"] == "Iceberg":
            # 冰山本体先记入“冰山表”（不是立即入 LOB）
            order_id = new_order["ID"]
            self.iceberg_dict[new_order["Side"]].addOrder(new_order)
            # 立即取出“第一份显示量”并放入 LOB（这样后续撮合才有可见量）
            iceberg_order, hit_iceberg = self.checkForIcebergs(new_order["Side"], order_id)
            self.insertOrderIntoLOB(iceberg_order)  # 这里的 iceberg_order 是“Limit”，会走到 super()
            return 
        # 其它类型（Limit/Market/…）按父类逻辑入簿
        super().insertOrderIntoLOB(new_order)
    
    def checkNewOrder(self, new_order):
        if new_order["Type"] == "Stop":
            # 止损单的合法性校验：需要“对手盘存在”，并且“当前不应立即触发”
            matchable = self.checkIfMatch(new_order)  # 用止损价当作“测试用限价”判断是否跨价差
            if len(self.LOB_dict[new_order["OpposingSide"]].orders["ID"]) == 0:
                # 对手盘没有挂单：无法判断触发关系，视为非法
                raise ValueError("Invalid stop order - No opposing orders")

            # 等价价位也视为“会触发”，因此也应视作非法（止损价必须“严格在远端”）
            equal_price = new_order["Price"] == self.LOB_dict[new_order["OpposingSide"]].orders["Price"][0]

            # —— 只要“现在就会触发”（包含相等），都是非法的止损价格 —— 
            if matchable or equal_price:
                raise ValueError("Invalid stop price")
            else:
                # 合法：把止损单加入它的“止损表”（稍后在撮合完成后检查是否触发）
                self.stop_dict[new_order["Side"]].addOrder(new_order)
            # 止损单本身不参与当前撮合，标记为 finished
            return new_order, True

        # 其它类型沿用父类（SeveralOrderMatchingAlgorithm）的预处理逻辑
        return super().checkNewOrder(new_order)
        
    def processNewOrder(self, new_order):
        # 先走标准的新单处理（包含：加 OpposingSide、处理 Market/IOC/ALO 等、随后撮合）
        super().processNewOrder(new_order)
        # 撮合完成后，检查是否有止损被触发（注意：规范要求“在触发它的订单撮合完成之后”再触发）
        self.onMatchingCompletion(new_order)

    def onMatchingCompletion(self, new_order):
        # 若该侧没有任何止损待触发，直接返回
        if len(self.stop_dict[new_order["Side"]].orders["ID"]) == 0:
            return
        
        # 取“最靠近触发点”的止损价（因已按买：低→高 / 卖：高→低排序，位于索引 0）
        best_stop_price = self.stop_dict[new_order["Side"]].orders["Price"][0]

        # 构造一个“测试订单对象”来询问：以该止损价为“限价”，此刻是否会跨价差
        temp_order = {"OpposingSide":new_order["OpposingSide"],  # 对手盘与新单相同
                      "Price": best_stop_price, "Type":"Limit"}  # 用“限价”做测试（是否会成交）

        matchable = self.checkIfMatch(temp_order)  # True 表示“已越过/等于”触发价
        # 防御：对手盘可能为空，此时“相等”无意义；否则比较是否“等价触发”
        equal_prices = (len(self.LOB_dict[new_order["OpposingSide"]].orders["ID"]) > 0 and
                        best_stop_price == self.LOB_dict[new_order["OpposingSide"]].orders["Price"][0])

        # —— 若“已越过/等于”触发价：把该止损单转为一张“市价单”并立刻处理 —— 
        if matchable or equal_prices:
            same_side = new_order["Side"]                                 # 止损方向与同侧一致
            stop_id   = self.stop_dict[same_side].orders["ID"][0]         # 止损单 ID
            stop_qty  = self.stop_dict[same_side].orders["Quantity"][0]   # 止损单数量
            current_time = dt.datetime.now().replace(second=0, microsecond=0)

            # 从止损表删除这条记录（避免重复触发）
            self.stop_dict[same_side].deleteOrder(0)

            # 构造并“提交”对应的市价单；价格字段仅作占位（市价单不用价格）
            market_order = {"Side": same_side, "OpposingSide": new_order["OpposingSide"],
                            "Type":"Market", "ID": stop_id, "Price": best_stop_price,
                            "Quantity": stop_qty, "Time": current_time}

            # 递归调用：把该“触发的市价单”按正常流程处理（可继续触发更多止损）
            self.processNewOrder(market_order)
        # 若未触发任何止损，什么也不做（返回）


# %%
def GetFullMatching(buy_dict, sell_dict, matching_rules, minimum_price_var = 1,
                 buy_iceberg_dict = None, sell_iceberg_dict = None, buy_stop_dict = None, sell_stop_dict = None):
    LOB_buy = LOBSide(side = "Buy", init_dict = buy_dict)
    LOB_sell = LOBSide(side = "Sell", init_dict = sell_dict)
    FullMatchingAlgo = AllOrderMatchingAlgorithm(LOB_buy, LOB_sell, matching_rules, 
                                                 minimum_price_var= minimum_price_var,
                                                 buy_iceberg_dict=buy_iceberg_dict, 
                                                 sell_iceberg_dict=sell_iceberg_dict,
                                                 buy_stop_dict=buy_stop_dict,
                                                 sell_stop_dict=sell_stop_dict)
    return FullMatchingAlgo

# Testing our modifications haven't broken the previous unit tests.
runner.run(TopLMMProRataMatchingClassSuite())



# %%
# Testing out the orders with immediate effects part (run the next code cell to load these)
runner.run(ImmediateOrderMatchingClassSuite())



# %%
# Testing out the orders with delayed effects part (run the last code cell to load these)
runner.run(DelayedOrderMatchingClassSuite())

# %%
class TestImmediateOrderMatching(unittest.TestCase):
    # Market order tests
    def test_ImmediateOrderMatchingAlgorithm_smallMarketOrder_matchesSomeOrders(self):
        buy_dict = {"ID":[1,2,3,4,5,6], 
                    "Price":[104,103,102,101,101,100], 
                    "Quantity":[10,20,40,40,10,20], 
                    "Time":[dt.datetime(2024,9,1,10,42,0),dt.datetime(2024,9,1,10,43,0),dt.datetime(2024,9,1,10,44,0),
                            dt.datetime(2024,9,1,10,45,0),dt.datetime(2024,9,1,10,46,0),dt.datetime(2024,9,1,10,47,0)]}
        sell_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}
        matching_rules = "FIFO"
        new_order = {"ID": 7, "Side":"Sell", "Type":"Market", "Quantity":71, "Time":dt.datetime(2024,9,1,10,49,0)}
        expected_buy_dict = {"ID":[4,5,6], 
                             "Price":[101,101,100], 
                             "Quantity":[39,10,20], 
                             "Time":[dt.datetime(2024,9,1,10,45,0),dt.datetime(2024,9,1,10,46,0),
                                     dt.datetime(2024,9,1,10,47,0)]}
        expected_sell_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}

        FullMatchingAlgo = GetFullMatching(buy_dict, sell_dict, matching_rules)
        FullMatchingAlgo.processNewOrder(new_order)
        new_buy_dict, new_sell_dict = FullMatchingAlgo.getLOBDicts()

        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict, expected_buy_dict)
    
    def test_ImmediateOrderMatchingAlgorithm_largeMarketOrder_matchesAllOrders(self):
        buy_dict = {"ID":[1,2,3,4,5,6], 
                    "Price":[104,103,102,101,101,100], 
                    "Quantity":[10,20,40,40,10,20], 
                    "Time":[dt.datetime(2024,9,1,10,42,0),dt.datetime(2024,9,1,10,43,0),dt.datetime(2024,9,1,10,44,0),
                            dt.datetime(2024,9,1,10,45,0),dt.datetime(2024,9,1,10,46,0),dt.datetime(2024,9,1,10,47,0)]}
        sell_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}
        matching_rules = "FIFO"
        new_order = {"ID": 7, "Side":"Sell", "Type":"Market", "Quantity":141, "Time":dt.datetime(2024,9,1,10,49,0)}
        expected_buy_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}
        expected_sell_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}

        FullMatchingAlgo = GetFullMatching(buy_dict, sell_dict, matching_rules)
        FullMatchingAlgo.processNewOrder(new_order)
        new_buy_dict, new_sell_dict = FullMatchingAlgo.getLOBDicts()
        
        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict, expected_buy_dict)
    
    def test_ImmediateOrderMatchingAlgorithm_marketOrderAgainstNothing_noChange(self):
        buy_dict ={"ID":[],"Price":[],"Quantity":[], "Time":[]}
        sell_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}
        matching_rules = "FIFO"
        new_order = {"ID": 7, "Side":"Sell", "Type":"Market", "Quantity":141, "Time":dt.datetime(2024,9,1,10,49,0)}
        expected_buy_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}
        expected_sell_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}

        FullMatchingAlgo = GetFullMatching(buy_dict, sell_dict, matching_rules)
        FullMatchingAlgo.processNewOrder(new_order)
        new_buy_dict, new_sell_dict = FullMatchingAlgo.getLOBDicts()
        
        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict, expected_buy_dict)
    
    # IOC tests:
    def test_ImmediateOrderMatchingAlgorithm_validIOC_completesImmediately(self):
        buy_dict ={"ID":[],"Price":[],"Quantity":[], "Time":[]}
        sell_dict = {"ID":[3,4,2],"Price":[101,101,102],"Quantity":[40,45,55], 
                     "Time":[dt.datetime(2024,9,1,10,44,0), dt.datetime(2024,9,1,10,45,0),dt.datetime(2024,9,1,10,41,0)]}
        matching_rules = "FIFO"
        new_order = {"ID": 4, "Side":"Buy", "TimeType":"IOC", "Quantity":83, 
                     "Price":102, "Time":dt.datetime(2024,9,1,10,49,0)}
        # 40 + 45 = 85 > 83, so the order should not be cancelled.
        # Should leave 2 units in order 4, and all of order 2.
        
        expected_buy_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}
        expected_sell_dict = {"ID":[4,2],"Price":[101,102],"Quantity":[2,55], 
                              "Time":[dt.datetime(2024,9,1,10,45,0),dt.datetime(2024,9,1,10,41,0)]}

        FullMatchingAlgo = GetFullMatching(buy_dict, sell_dict, matching_rules)
        FullMatchingAlgo.processNewOrder(new_order)
        new_buy_dict, new_sell_dict = FullMatchingAlgo.getLOBDicts()
        
        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict, expected_buy_dict)
    
    def test_ImmediateOrderMatchingAlgorithm_partialIOC_completesImmediately(self):
        buy_dict ={"ID":[],"Price":[],"Quantity":[], "Time":[]}
        sell_dict = {"ID":[3,4,2],"Price":[101,101,102],"Quantity":[40,45,55], 
                     "Time":[dt.datetime(2024,9,1,10,44,0), dt.datetime(2024,9,1,10,45,0),dt.datetime(2024,9,1,10,41,0)]}
        matching_rules = "FIFO"
        new_order = {"ID": 4, "Side":"Buy", "TimeType":"IOC", "Quantity":100, 
                     "Price":102, "Time":dt.datetime(2024,9,1,10,49,0)}
        # 40 + 45 = 85 > 83, so the order should not be cancelled.
        # Should leave 2 units in order 4, and all of order 2.
        
        expected_buy_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}
        expected_sell_dict = {"ID":[2],"Price":[102],"Quantity":[55], 
                              "Time":[dt.datetime(2024,9,1,10,41,0)]}

        FullMatchingAlgo = GetFullMatching(buy_dict, sell_dict, matching_rules)
        FullMatchingAlgo.processNewOrder(new_order)
        new_buy_dict, new_sell_dict = FullMatchingAlgo.getLOBDicts()
        
        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict, expected_buy_dict)

    def test_ImmediateOrderMatchingAlgorithm_partialMarketIOC_completesImmediately(self):
        buy_dict ={"ID":[],"Price":[],"Quantity":[], "Time":[]}
        sell_dict = {"ID":[3,4,2],"Price":[101,101,102],"Quantity":[40,45,55], 
                     "Time":[dt.datetime(2024,9,1,10,44,0), dt.datetime(2024,9,1,10,45,0),dt.datetime(2024,9,1,10,41,0)]}
        matching_rules = "FIFO"
        new_order = {"ID": 4, "Side":"Buy", "Type":"Market", "TimeType":"IOC", "Quantity":100, 
                     "Time":dt.datetime(2024,9,1,10,49,0)}
        # 40 + 45 = 85 > 83, so the order should not be cancelled.
        # Should leave 2 units in order 4, and all of order 2.
        
        expected_buy_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}
        expected_sell_dict = {"ID":[2],"Price":[102],"Quantity":[55], 
                              "Time":[dt.datetime(2024,9,1,10,41,0)]}

        FullMatchingAlgo = GetFullMatching(buy_dict, sell_dict, matching_rules)
        FullMatchingAlgo.processNewOrder(new_order)
        new_buy_dict, new_sell_dict = FullMatchingAlgo.getLOBDicts()
        
        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict, expected_buy_dict)
    def test_ImmediateOrderMatchingAlgorithm_againstEmptyIOC_noChange(self):
        buy_dict ={"ID":[],"Price":[],"Quantity":[], "Time":[]}
        sell_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}
        matching_rules = "FIFO"
        new_order = {"ID": 4, "Side":"Buy", "Type":"Market", "TimeType":"IOC", "Quantity":100, 
                     "Time":dt.datetime(2024,9,1,10,49,0)}
        # 40 + 45 = 85 > 83, so the order should not be cancelled.
        # Should leave 2 units in order 4, and all of order 2.
        
        expected_buy_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}
        expected_sell_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}
        FullMatchingAlgo = GetFullMatching(buy_dict, sell_dict, matching_rules)
        FullMatchingAlgo.processNewOrder(new_order)
        new_buy_dict, new_sell_dict = FullMatchingAlgo.getLOBDicts()
        
        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict, expected_buy_dict)
    # ALO tests
    def test_ImmediateOrderMatchingAlgorithm_validALO_addsToLOB(self):
        buy_dict ={"ID":[],"Price":[],"Quantity":[], "Time":[]}
        sell_dict = {"ID":[3],"Price":[101],"Quantity":[40], "Time":[dt.datetime(2024,9,1,10,44,0)]}
        matching_rules = "FIFO"
        new_order = {"ID": 4, "Side":"Buy", "Type":"ALO", "Quantity":141, 
                     "Price":100, "Time":dt.datetime(2024,9,1,10,49,0)}
        expected_buy_dict = {"ID":[4],"Price":[100],"Quantity":[141], "Time":[dt.datetime(2024,9,1,10,49,0)]}
        expected_sell_dict = {"ID":[3],"Price":[101],"Quantity":[40], "Time":[dt.datetime(2024,9,1,10,44,0)]}

        FullMatchingAlgo = GetFullMatching(buy_dict, sell_dict, matching_rules)
        FullMatchingAlgo.processNewOrder(new_order)
        new_buy_dict, new_sell_dict = FullMatchingAlgo.getLOBDicts()
        
        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict, expected_buy_dict)
    
    def test_ImmediateOrderMatchingAlgorithm_invalidBuyALO_changesPrice(self):
        buy_dict ={"ID":[],"Price":[],"Quantity":[], "Time":[]}
        sell_dict = {"ID":[3],"Price":[101],"Quantity":[40], "Time":[dt.datetime(2024,9,1,10,44,0)]}
        matching_rules = "FIFO"
        new_order = {"ID": 4, "Side":"Buy", "Type":"ALO", "Quantity":141, 
                     "Price":108, "Time":dt.datetime(2024,9,1,10,49,0)}
        expected_buy_dict = {"ID":[4],"Price":[100],"Quantity":[141], "Time":[dt.datetime(2024,9,1,10,49,0)]}
        expected_sell_dict = {"ID":[3],"Price":[101],"Quantity":[40], "Time":[dt.datetime(2024,9,1,10,44,0)]}

        FullMatchingAlgo = GetFullMatching(buy_dict, sell_dict, matching_rules)
        FullMatchingAlgo.processNewOrder(new_order)
        new_buy_dict, new_sell_dict = FullMatchingAlgo.getLOBDicts()
        
        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict, expected_buy_dict)
    
    def test_ImmediateOrderMatchingAlgorithm_invalidSellALO_changesPrice(self):
        buy_dict = {"ID":[4],"Price":[100],"Quantity":[141], "Time":[dt.datetime(2024,9,1,10,49,0)]}
        sell_dict = {"ID":[3],"Price":[101],"Quantity":[40], "Time":[dt.datetime(2024,9,1,10,44,0)]}
        matching_rules = "FIFO"
        new_order = {"ID": 5, "Side":"Sell", "Type":"ALO", "Quantity":145, 
                     "Price":96, "Time":dt.datetime(2024,9,1,10,49,0)}
        expected_buy_dict = {"ID":[4],"Price":[100],"Quantity":[141], "Time":[dt.datetime(2024,9,1,10,49,0)]}
        expected_sell_dict = {"ID":[3,5],"Price":[101,101],"Quantity":[40,145], 
                              "Time":[dt.datetime(2024,9,1,10,44,0),dt.datetime(2024,9,1,10,49,0)]}

        FullMatchingAlgo = GetFullMatching(buy_dict, sell_dict, matching_rules)
        FullMatchingAlgo.processNewOrder(new_order)
        new_buy_dict, new_sell_dict = FullMatchingAlgo.getLOBDicts()
        
        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict, expected_buy_dict)
    
def ImmediateOrderMatchingClassSuite():

    loader = unittest.TestLoader()
    full_matching_suite = loader.loadTestsFromTestCase(TestImmediateOrderMatching)
    return full_matching_suite

# runner.run(ImmediateOrderMatchingClassSuite())

# %%
class TestDelayedOrderMatching(unittest.TestCase):
    # Stop orders:
    def test_DelayedOrderMatchingAlgorithm_invalidStopSellOrder_ValueError(self):
        buy_dict = {"ID":[1,2], 
                    "Price":[103,103], 
                    "Quantity":[10,20], 
                    "Time":[dt.datetime(2024,9,1,10,42,0),dt.datetime(2024,9,1,10,43,0)]}
        sell_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}

        matching_rules = "FIFO"
        new_order = {"ID":7, "Side":"Sell","Type":"Stop", "Price":103, "Quantity":50,"Time":dt.datetime(2024,9,1,10,48,0)}

        expected_exception = ValueError
        FullMatchingAlgo = GetFullMatching(buy_dict, sell_dict, matching_rules)
        def addOrder(new_order):
            FullMatchingAlgo.processNewOrder(new_order)
        self.assertRaises(expected_exception, addOrder ,new_order)
    
    def test_DelayedOrderMatchingAlgorithm_noOpposingStopOrder_ValueError(self):
        sell_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}
        buy_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}

        matching_rules = "FIFO"
        new_order = {"ID":7, "Side":"Buy","Type":"Stop", "Price":104, "Quantity":52,"Time":dt.datetime(2024,9,1,10,48,0)}
        # expected_sell_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}
        # expected_buy_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}
        # expected_stop_buy = {"ID":[7], "Price":[104], "Quantity":[52], "Time":[dt.datetime(2024,9,1,10,48,0)]}
        # expected_stop_sell = {"ID":[],"Price":[],"Quantity":[], "Time":[]}

        expected_exception = ValueError
        FullMatchingAlgo = GetFullMatching(buy_dict, sell_dict, matching_rules)
        def addOrder(new_order):
            FullMatchingAlgo.processNewOrder(new_order)
        self.assertRaises(expected_exception, addOrder ,new_order)

    def test_DelayedOrderMatchingAlgorithm_validStopSellOrder_addsToStopOrderDict(self):
        buy_dict = {"ID":[1,2], 
                    "Price":[103,103], 
                    "Quantity":[10,20], 
                    "Time":[dt.datetime(2024,9,1,10,42,0),dt.datetime(2024,9,1,10,43,0)]}
        sell_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}

        matching_rules = "FIFO"
        new_order = {"ID":7, "Side":"Sell","Type":"Stop", "Price":100, "Quantity":50,"Time":dt.datetime(2024,9,1,10,48,0)}
        expected_buy_dict = {"ID":[1,2], 
                    "Price":[103,103], 
                    "Quantity":[10,20], 
                    "Time":[dt.datetime(2024,9,1,10,42,0),dt.datetime(2024,9,1,10,43,0)]}
        expected_sell_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}
        expected_stop_buy = {"ID":[], "Price":[], "Quantity":[], "Time":[]}
        expected_stop_sell = {"ID":[7], "Price":[100], "Quantity":[50], "Time":[dt.datetime(2024,9,1,10,48,0)]}

        FullMatchingAlgo = GetFullMatching(buy_dict, sell_dict, matching_rules)
        FullMatchingAlgo.processNewOrder(new_order)
        new_buy_dict, new_sell_dict = FullMatchingAlgo.getLOBDicts()
        new_stop_buy = FullMatchingAlgo.stop_dict["Buy"].orders
        new_stop_sell = FullMatchingAlgo.stop_dict["Sell"].orders

        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict, expected_buy_dict)
        self.assertDictEqual(new_stop_buy, expected_stop_buy)
        self.assertDictEqual(new_stop_sell, expected_stop_sell)

    def test_DelayedOrderMatchingAlgorithm_validStopBuyOrder_addsToStopOrderDict(self):
        sell_dict = {"ID":[1,2], 
                    "Price":[103,103], 
                    "Quantity":[10,20], 
                    "Time":[dt.datetime(2024,9,1,10,42,0),dt.datetime(2024,9,1,10,43,0)]}
        buy_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}

        matching_rules = "FIFO"
        new_order = {"ID":7, "Side":"Buy","Type":"Stop", "Price":104, "Quantity":52,"Time":dt.datetime(2024,9,1,10,48,0)}
        expected_sell_dict = {"ID":[1,2], 
                    "Price":[103,103], 
                    "Quantity":[10,20], 
                    "Time":[dt.datetime(2024,9,1,10,42,0),dt.datetime(2024,9,1,10,43,0)]}
        expected_buy_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}
        expected_stop_buy = {"ID":[7], "Price":[104], "Quantity":[52], "Time":[dt.datetime(2024,9,1,10,48,0)]}
        expected_stop_sell = {"ID":[],"Price":[],"Quantity":[], "Time":[]}

        FullMatchingAlgo = GetFullMatching(buy_dict, sell_dict, matching_rules)
        FullMatchingAlgo.processNewOrder(new_order)
        new_buy_dict, new_sell_dict = FullMatchingAlgo.getLOBDicts()
        new_stop_buy = FullMatchingAlgo.stop_dict["Buy"].orders
        new_stop_sell = FullMatchingAlgo.stop_dict["Sell"].orders

        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict, expected_buy_dict)
        self.assertDictEqual(new_stop_buy, expected_stop_buy)
        self.assertDictEqual(new_stop_sell, expected_stop_sell)
    
    # Converting stop orders into market orders:
    def test_DelayedOrderMatchingAlgorithm_triggerStopBuyOrder_processesMarketOrder(self):
        sell_dict = {"ID":[1,2], 
                    "Price":[101,105], 
                    "Quantity":[10,20], 
                    "Time":[dt.datetime(2024,9,1,10,42,0),dt.datetime(2024,9,1,10,43,0)]}
        buy_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}

        matching_rules = "FIFO"
        
        buy_stop_dict = {"ID":[6],"Price":[104],"Quantity":[10], "Time":[dt.datetime(2024,9,1,10,48,0)]}
        # We place an order, price 102, that should match with order 1 and then be added to the LOB.
        # It should then trigger the stop buy order to send a market order.
        new_order = {"ID":7, "Side":"Buy","Type":"Limit", "Price":101, "Quantity":30,"Time":dt.datetime(2024,9,1,10,48,0)}

        # This stop buy order places a buy market order, quantity 10, which matches with half of order 2.
        # Because the stop price is 105, and after the order completes the best ask is 105.
        expected_sell_dict = {"ID":[2], 
                    "Price":[105], 
                    "Quantity":[10], 
                    "Time":[dt.datetime(2024,9,1,10,43,0)]}
        
        expected_buy_dict = {"ID":[7],"Price":[101],"Quantity":[20], "Time":[dt.datetime(2024,9,1,10,48,0)]}
        expected_stop_buy = {"ID":[], "Price":[], "Quantity":[], "Time":[]}
        expected_stop_sell = {"ID":[],"Price":[],"Quantity":[], "Time":[]}

        FullMatchingAlgo = GetFullMatching(buy_dict, sell_dict, matching_rules, buy_stop_dict = buy_stop_dict)
        FullMatchingAlgo.processNewOrder(new_order)
        new_buy_dict, new_sell_dict = FullMatchingAlgo.getLOBDicts()
        new_stop_buy = FullMatchingAlgo.stop_dict["Buy"].orders
        new_stop_sell = FullMatchingAlgo.stop_dict["Sell"].orders

        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict, expected_buy_dict)
        self.assertDictEqual(new_stop_buy, expected_stop_buy)
        self.assertDictEqual(new_stop_sell, expected_stop_sell)  

    def test_DelayedOrderMatchingAlgorithm_triggerStopSellOrderCascade_processesMarketOrders(self):
        buy_dict = {"ID":[1,2,3,4],
                    "Price":[101,100,98,95],
                    "Quantity":[10,40,50,35],
                    "Time":[dt.datetime(2024,9,1,10,42,0),dt.datetime(2024,9,1,10,43,0),
                            dt.datetime(2024,9,1,10,45,0),dt.datetime(2024,9,1,10,47,0)]}
        sell_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}
    
        matching_rules = "FIFO"
        sell_stop_dict = {"ID":[5,6,7,8], "Price":[100,100,99,94], "Quantity":[45,60,10,100], 
                          "Time":[dt.datetime(2024,9,1,10,48,0), dt.datetime(2024,9,1,10,49,0),
                                  dt.datetime(2024,9,1,10,51,0), dt.datetime(2024,9,1,10,51,0)]}
        # We place an order, price 101, which will match with order 1 completely. 
        new_order = {"ID":9,"Side":"Sell", "Type":"Limit", "Price":101, "Quantity":10, 
                     "Time": dt.datetime(2024,9,1,10,53,0)}
        # This reduces the bid price to 100, and then the first stop order, order 5, triggers and becomes a market order.
        # This matches with order 2 completely, and 5 of order 3, leaving 45.
        # Order 3, the best bid, has price 98, which is lower than 100 triggering order 6 to become a market order.
        # This matches with all of order 3, and 15 of order 4, leaving 20.

        # Now the best bid price is 95, and order 7 triggers as 95<99. This becomes a market order
        # and matches with order 4, leaving 10 in order 4.

        # We can see here that several stop orders have lead to a rapid drop in price, 
        # and a fairly conservative 99 stop order was fulfilled at 95, which is far from ideal.

        expected_sell_dict = {"ID":[], "Price":[], "Quantity":[], "Time":[]}
        expected_stop_buy = {"ID":[], "Price":[], "Quantity":[], "Time":[]}
        expected_stop_sell = {"ID":[8], "Price":[94], "Quantity":[100], "Time":[dt.datetime(2024,9,1,10,51,0)]}
        expected_buy_dict = {"ID":[4], 
                    "Price":[95], 
                    "Quantity":[10], 
                    "Time":[dt.datetime(2024,9,1,10,47,0)]}
        
        FullMatchingAlgo = GetFullMatching(buy_dict, sell_dict, matching_rules, sell_stop_dict = sell_stop_dict)
        FullMatchingAlgo.processNewOrder(new_order)
        new_buy_dict, new_sell_dict = FullMatchingAlgo.getLOBDicts()
        new_stop_buy = FullMatchingAlgo.stop_dict["Buy"].orders
        new_stop_sell = FullMatchingAlgo.stop_dict["Sell"].orders

        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict, expected_buy_dict)
        self.assertDictEqual(new_stop_buy, expected_stop_buy)
        self.assertDictEqual(new_stop_sell, expected_stop_sell)  
    
    # Iceberg orders:
    def test_DelayedOrderMatchingAlgorithm_newIcebergOrder_addsIcebergOrder(self):
        buy_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}
        sell_dict = {"ID":[2],"Price":[99],"Quantity":[20], "Time":[dt.datetime(2024,9,1,10,47,0)]}
        new_order = {"ID":1, "Side":"Buy","Type":"Iceberg",
                     "Price":100, "Quantity":100, "DisplayQuantity":20, "Time":dt.datetime(2024,9,1,10,51,0)}
        matching_rules = "FIFO"
        FullMatchingAlgo = GetFullMatching(buy_dict, sell_dict, matching_rules)
        FullMatchingAlgo.processNewOrder(new_order)
        new_buy_dict, new_sell_dict = FullMatchingAlgo.getLOBDicts()
        new_iceberg_buy = FullMatchingAlgo.iceberg_dict["Buy"].orders
        new_iceberg_sell = FullMatchingAlgo.iceberg_dict["Sell"].orders
        expected_sell_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}
        current_time = dt.datetime.now().replace(second=0, microsecond=0)
        expected_buy_dict = {"ID":[1], "Price":[100], "Quantity":[20], "Time":[current_time]}
        expected_iceberg_buy = {"ID":[1], "Price":[100], "Quantity":[60], "DisplayQuantity":[20],
                                "Time":[dt.datetime(2024,9,1,10,51,0)]}
        expected_iceberg_sell = {"ID":[],"Price":[],"Quantity":[],"DisplayQuantity":[], "Time":[]}
        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict, expected_buy_dict)
        self.assertDictEqual(new_iceberg_buy, expected_iceberg_buy)
        self.assertDictEqual(new_iceberg_sell, expected_iceberg_sell)
    
    def test_DelayedOrderMatchingAlgorithm_oldIcebergBuyOrder_addsLimitOrderWhenHit(self):
        buy_dict = {"ID":[1],"Price":[100],"Quantity":[50], "Time":[dt.datetime(2024,9,1,10,52,0)]}
        sell_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}
        buy_iceberg_dict = {"ID":[1],"Price":[100], "Quantity":[150], 
                            "DisplayQuantity":[50], "Time":[dt.datetime(2024,9,1,10,51,0)]}
        new_order = {"ID":2, "Side":"Sell", "Price":99, "Quantity":67, 
                     "Time":dt.datetime(2024,9,1,10,51,0)}

        matching_rules = "FIFO"
        FullMatchingAlgo = GetFullMatching(buy_dict, sell_dict, matching_rules, buy_iceberg_dict=buy_iceberg_dict)
        FullMatchingAlgo.processNewOrder(new_order)
        new_buy_dict, new_sell_dict = FullMatchingAlgo.getLOBDicts()
        new_iceberg_buy = FullMatchingAlgo.iceberg_dict["Buy"].orders
        new_iceberg_sell = FullMatchingAlgo.iceberg_dict["Sell"].orders

        expected_sell_dict = {"ID":[],"Price":[],"Quantity":[], "Time":[]}
        current_time = dt.datetime.now().replace(second=0, microsecond=0)
        expected_buy_dict = {"ID":[1], "Price":[100], "Quantity":[33], "Time":[current_time]}
        expected_iceberg_buy = {"ID":[1], "Price":[100], "Quantity":[100], "DisplayQuantity":[50],
                                "Time":[dt.datetime(2024,9,1,10,51,0)]}
        expected_iceberg_sell = {"ID":[],"Price":[],"Quantity":[],"DisplayQuantity":[], "Time":[]}
        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict, expected_buy_dict)
        self.assertDictEqual(new_iceberg_buy, expected_iceberg_buy)
        self.assertDictEqual(new_iceberg_sell, expected_iceberg_sell)
    
    def test_DelayedOrderMatchingAlgorithm_oldIcebergSellOrder_addsLimitOrderWhenHit(self):
        sell_dict = {"ID":[1],"Price":[100],"Quantity":[50], "Time":[dt.datetime(2024,9,1,10,52,0)]}
        buy_dict = {"ID":[], "Price":[],"Quantity":[], "Time":[]}
        sell_iceberg_dict = {"ID":[1],"Price":[100], "Quantity":[35], 
                            "DisplayQuantity":[50], "Time":[dt.datetime(2024,9,1,10,51,0)]}
        
        new_order = {"ID":2, "Side":"Buy", "Price":101, "Quantity":67, 
                     "Time":dt.datetime(2024,9,1,10,51,0)}

        matching_rules = "FIFO"
        FullMatchingAlgo = GetFullMatching(buy_dict, sell_dict, matching_rules, sell_iceberg_dict=sell_iceberg_dict)
        FullMatchingAlgo.processNewOrder(new_order)
        new_buy_dict, new_sell_dict = FullMatchingAlgo.getLOBDicts()
        new_iceberg_buy = FullMatchingAlgo.iceberg_dict["Buy"].orders
        new_iceberg_sell = FullMatchingAlgo.iceberg_dict["Sell"].orders

        
        current_time = dt.datetime.now().replace(second=0, microsecond=0)
        expected_sell_dict = {"ID":[1],"Price":[100],"Quantity":[18], "Time":[current_time]}
        expected_buy_dict = {"ID":[], "Price":[], "Quantity":[], "Time":[]}
        expected_iceberg_buy = {"ID":[],"Price":[],"Quantity":[],"DisplayQuantity":[], "Time":[]}
        expected_iceberg_sell = {"ID":[],"Price":[],"Quantity":[],"DisplayQuantity":[], "Time":[]}

        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict, expected_buy_dict)
        self.assertDictEqual(new_iceberg_buy, expected_iceberg_buy)
        self.assertDictEqual(new_iceberg_sell, expected_iceberg_sell)
    
    def test_DelayedOrderMatchingAlgorithm_partialMatchIcebergBuyOrder_matchesThenIcebergs(self):
        sell_dict = {"ID":[1],"Price":[100],"Quantity":[45], "Time":[dt.datetime(2024,9,1,10,52,0)]}
        buy_dict = {"ID":[], "Price":[],"Quantity":[], "Time":[]}
        sell_iceberg_dict = {"ID":[1],"Price":[100], "Quantity":[35], 
                            "DisplayQuantity":[50], "Time":[dt.datetime(2024,9,1,10,51,0)]}
        
        new_order = {"ID":3, "Side":"Buy", "Type":"Iceberg", "Price":101, "Quantity":200, "DisplayQuantity":50,
                     "Time":dt.datetime(2024,9,1,10,51,0)}

        matching_rules = "FIFO"
        FullMatchingAlgo = GetFullMatching(buy_dict, sell_dict, matching_rules, sell_iceberg_dict=sell_iceberg_dict)
        FullMatchingAlgo.processNewOrder(new_order)
        new_buy_dict, new_sell_dict = FullMatchingAlgo.getLOBDicts()
        new_iceberg_buy = FullMatchingAlgo.iceberg_dict["Buy"].orders
        new_iceberg_sell = FullMatchingAlgo.iceberg_dict["Sell"].orders

        current_time = dt.datetime.now().replace(second=0, microsecond=0)
        expected_sell_dict = {"ID":[], "Price":[], "Quantity":[], "Time":[]}
        expected_buy_dict = {"ID":[3],"Price":[101],"Quantity":[50], "Time":[current_time]}
        expected_iceberg_buy = {"ID":[3],"Price":[101],"Quantity":[70],"DisplayQuantity":[50],
                                "Time":[dt.datetime(2024,9,1,10,51,0)]}
        expected_iceberg_sell = {"ID":[],"Price":[],"Quantity":[],"DisplayQuantity":[], "Time":[]}

        self.assertDictEqual(new_sell_dict, expected_sell_dict)
        self.assertDictEqual(new_buy_dict, expected_buy_dict)
        self.assertDictEqual(new_iceberg_buy, expected_iceberg_buy)
        self.assertDictEqual(new_iceberg_sell, expected_iceberg_sell)
    
def DelayedOrderMatchingClassSuite():
    loader = unittest.TestLoader()
    full_matching_suite = loader.loadTestsFromTestCase(TestDelayedOrderMatching)
    return full_matching_suite

# runner.run(DelayedOrderMatchingClassSuite())s

# %% [markdown]
# # Unit3

# %% [markdown]
# # 4 - The current fair price for an asset

# %%
import pandas as pd
import numpy as np
import plotly.express as px

# %%
# We download the data in the above format:
def get_sample_dataframe(name):
    file_name = name + "_sample.csv"
    file_path = r"C:\Users\sxmxs\Desktop\Limit-Order-Books-main\data\\"
    colnames = ["Date","Time","BidPrice","BidQuantity","AskPrice","AskQuantity"]
    sample_df = pd.read_csv(file_path + file_name, header = None, names=colnames)
    # Create a datetime column combining the time and date.
    datetime = sample_df["Date"] + " " + sample_df["Time"]
    sample_df["Datetime"] = pd.to_datetime(datetime, format = "%d/%m/%Y %H:%M:%S")
    return sample_df
sample_df = get_sample_dataframe("small")
sample_df

# %%
def add_mid(LOB_dataframe, inplace = False):
    """
    在 LOB 数据中新增一列 'MidPrice'，按 (AskPrice + BidPrice) / 2 计算。
    参数
    ----
    LOB_dataframe : pandas.DataFrame
        必须包含列 'BidPrice' 与 'AskPrice'
    inplace : bool
        False 时返回拷贝；True 时在原地修改
    """
    # 根据 inplace 选择是拷贝还是直接修改原表
    new_df = LOB_dataframe if inplace else LOB_dataframe.copy()

    # 确保参与计算的两列是数值型（防止读入为字符串引发拼接）
    new_df["BidPrice"] = pd.to_numeric(new_df["BidPrice"], errors="coerce")
    new_df["AskPrice"] = pd.to_numeric(new_df["AskPrice"], errors="coerce")

    # 核心：按公式计算中间价，并生成新列 'MidPrice'
    new_df["MidPrice"] = (new_df["AskPrice"] + new_df["BidPrice"]) / 2.0

    return new_df
    
add_mid(sample_df, inplace = True).head(5)

# %%
def plot_price_estimates(dataframe):
    """
    将所有以 'Price' 结尾的列与 Datetime 一起绘制为时间序列折线图。
    例如：BidPrice / AskPrice / MidPrice。
    """
    # 1) 找出所有价格列名
    column_names = dataframe.columns.values.tolist()
    price_names = [col_name for col_name in column_names if col_name.endswith("Price")]

    # 2) 仅保留用得上的列，先按时间排一下序（可选，画面更平滑）
    df_plot = dataframe.sort_values("Datetime")[["Datetime"] + price_names].copy()

    # 3) 宽表 -> 长表：每一行是 (Datetime, 价格列名, 对应价格值)
    #    var_name 给价格列一个类别名；value_name 为具体价格数值
    long_df = pd.melt(
        df_plot,
        id_vars=["Datetime"],
        value_vars=price_names,
        var_name="PriceType",
        value_name="Price"
    )

    # 4) 画图：横轴 Datetime，纵轴 Price，不同价格类型用不同颜色
    fig = px.line(
        long_df,
        x="Datetime",
        y="Price",
        color="PriceType",
        labels={"Datetime": "Datetime", "Price": "Price", "PriceType": "Series"}
    )

    return fig


# %%
fig = plot_price_estimates(sample_df)
fig.update_layout(title = "Bid, Ask and Mid prices")
fig.show()

# %% [markdown]
# # 4.1.2 - The weighted mid price

# %%
sample_df.head(2)

# %%
def add_imabalance(LOB_dataframe, inplace = False):
    if not inplace:
        new_df = LOB_dataframe.copy()
    else:
        new_df = LOB_dataframe
    I_bar = (new_df["BidQuantity"] - new_df["AskQuantity"])/ (new_df["AskQuantity"] + new_df["BidQuantity"])
    new_df["Imbalance"] = I_bar
    return new_df

# %%
def add_weighted_mid(LOB_dataframe, inplace = False):
    """
    在 LOB 数据中新增一列 'WeightedMidPrice'（加权中间价）。
    若数据框中不存在 'Imbalance'，会自动调用 add_imabalance 先计算失衡度。
    公式： P_WM = P_M + (AskPrice - BidPrice) * (Imbalance / 2)
    其中 Imbalance = (Q^b - Q^a) / (Q^b + Q^a)。
    """
    # 1) 根据 inplace 决定是否拷贝
    new_df = LOB_dataframe if inplace else LOB_dataframe.copy()

    # 2) 确保价格列是数值型，避免字符串相加
    new_df["BidPrice"] = pd.to_numeric(new_df["BidPrice"], errors="coerce")
    new_df["AskPrice"] = pd.to_numeric(new_df["AskPrice"], errors="coerce")

    # 3) 若没有 Imbalance 列，则先计算（调用上面已给的 add_imabalance）
    if "Imbalance" not in new_df.columns:
        # 这里用 inplace=True 直接在 new_df 上添加列，避免再返回赋值
        add_imabalance(new_df, inplace=True)

    # 4) 准备 Mid 和 Spread。若已存在 MidPrice 可直接用；否则现场计算。
    if "MidPrice" in new_df.columns:
        mid = new_df["MidPrice"]
    else:
        mid = (new_df["AskPrice"] + new_df["BidPrice"]) / 2.0
        new_df["MidPrice"] = mid  # 顺手补上，后续画图也会用到

    spread = new_df["AskPrice"] - new_df["BidPrice"]

    # 5) 根据等价式：P_WM = P_M + spread * (Imbalance / 2)
    new_df["WeightedMidPrice"] = mid + spread * (new_df["Imbalance"] / 2.0)

    return new_df
    
add_weighted_mid(sample_df, inplace = True).head(5)

# %%
fig = plot_price_estimates(sample_df)
fig.update_layout(title = "Bid, Ask, Mid and Weighted Mid prices")
fig.show()

# %%
sample_df.tail(2)

# %%


# %%
# Loading the Bank of America dataframe
BAC_df = get_sample_dataframe("BAC")

# %%
# Computing our price estimates
add_mid(BAC_df, inplace = True)
add_weighted_mid(BAC_df, inplace = True)
BAC_df.shape


# %%
BAC_df.head(5)

# %%
# 101 points evenly spaced from -1 to 1
imb_bins = np.linspace(-1, 1, 101) 
# We split up our imbalances using pd.cut. In our case, this function takes in:
# - The column we wish to convert into bins
# - bins = The edges of each bin
# - labels to call the bins
bin_midpoints = (imb_bins[1:] + imb_bins[:-1])/2
BAC_df["BinnedImbalance"] = pd.cut(BAC_df["Imbalance"], bins = imb_bins, labels = bin_midpoints)

# Next we compute the average change in price within each bin:
print(BAC_df.groupby("BinnedImbalance", observed = True)["MidPrice"].mean().head(5))
# This computes the mean mid price for each bin

# %%
# However, we want the change in the Mid price, so we write:
BAC_df["MidPriceChange"] = BAC_df["MidPrice"].pct_change().shift(-1)
# Shift -1 pushes it into the future, so we get the change in the mid price at the next step.
BAC_df.groupby("BinnedImbalance", observed=True)["MidPriceChange"].mean()

# %%
# Can then store this as a new column:
grouped_averages = BAC_df.groupby("BinnedImbalance", observed=True)["MidPriceChange"].mean()
imbalances = grouped_averages.keys()
avg_mid_price_change = grouped_averages.values
plot_df = pd.DataFrame({"Imbalance":imbalances, "AvgMidPriceChange":avg_mid_price_change})
# # Now we can plot this against the Binned Imbalance:
px.line(plot_df, x="Imbalance",y="AvgMidPriceChange")

# %%
def plot_price_change(dataframe):
    """
    对所有“估计价格”列（以 Price 结尾，但不含 Bid/Ask）：
    - 按失衡度（Imbalance）分箱（[-1,1] 100 桶）
    - 计算该估计价格“下一步百分比变化”的分箱均值
    - 在同一张图上画多条曲线进行对比

    不修改原始 dataframe。
    """
    # 1) 复制一份，保证对原数据“只读”
    df = dataframe.copy()

    # 2) 若缺少 Imbalance，则先计算（使用你之前提供的 add_imabalance）
    if "Imbalance" not in df.columns:
        add_imabalance(df, inplace=True)

    # 3) 找出“估计类价格”列：名称以 "Price" 结尾，但排除 Bid/Ask
    all_cols = df.columns.tolist()
    est_price_cols = [c for c in all_cols
                      if c.endswith("Price") and c not in ("BidPrice", "AskPrice")]

    # 若没有可画的估计列，直接给出空图以避免异常
    if len(est_price_cols) == 0:
        return px.line(title="No estimated price columns found.")

    # 4) 失衡度分箱（与前文保持一致：[-1,1] 等宽 100 桶）
    imb_bins = np.linspace(-1, 1, 101)
    bin_midpoints = (imb_bins[1:] + imb_bins[:-1]) / 2
    df["BinnedImbalance"] = pd.cut(df["Imbalance"],
                                   bins=imb_bins,
                                   labels=bin_midpoints)

    # 5) 对每个估计价格，计算“下一步百分比变化”并在各个分箱内取均值
    pieces = []
    for col in est_price_cols:
        # 下一步的相对变化：pct_change 后再 shift(-1) 把变化对齐到“当前时刻的失衡度”
        next_step_change = df[col].pct_change().shift(-1)

        # 按分箱求均值；observed=True 只保留实际出现过的桶
        grouped = next_step_change.groupby(df["BinnedImbalance"], observed=True).mean()

        # 组装为一个可拼接的小表
        part = pd.DataFrame({
            "Imbalance": grouped.index.astype(float),      # x 轴使用分箱的中点
            "AvgChange": grouped.values,                   # y 轴为“下一步平均变化”
            "Estimator": col.replace("Price", "")          # 图例：去掉 "Price"
        })
        pieces.append(part)

    # 6) 合并所有估计方法的结果，转成长表后画图
    plot_df = pd.concat(pieces, ignore_index=True)

    fig = px.line(
        plot_df,
        x="Imbalance",
        y="AvgChange",
        color="Estimator",
        labels={
            "Imbalance": "Imbalance",
            "AvgChange": "Average price change"   # y 轴名
        }
    )
    # 图例标题
    fig.update_layout(legend_title_text="Estimation type",
                      title="Avg next-step price change vs. Imbalance")

    return fig


# %%
plot_price_change(BAC_df)

# %% [markdown]
# # 4.1.4 - Non linear weighted mid prices

# %%
def f(x):
    return -0.612745 * x**5+0.553513 *x**3+1.05923* x
def add_NLWMPrice(dataframe, inplace = False):
    if not inplace:
        new_df = dataframe.copy()
    else:
        new_df = dataframe
    mid = (new_df["BidPrice"] + new_df["AskPrice"])/2
    spread = new_df["AskPrice"] - new_df["BidPrice"]
    if "Imbalance" not in new_df.columns:
        new_df = add_imabalance(new_df, inplace=True)
    new_df["Non-Linear WeightedMidPrice"] = mid + spread * f(new_df["Imbalance"])/2
    return new_df

# %%
plot_price_change(add_NLWMPrice(BAC_df))
# add_NLWMPrice(BAC_df)

# %%
CVX_df = get_sample_dataframe("BAC")
add_mid(CVX_df, inplace = True)
add_weighted_mid(CVX_df, inplace = True)
add_NLWMPrice(CVX_df, inplace = True)

plot_price_change(CVX_df)

# %%
CVX_df = get_sample_dataframe("CVX")
add_mid(CVX_df, inplace = True)
add_weighted_mid(CVX_df, inplace = True)
add_NLWMPrice(CVX_df, inplace = True)

plot_price_change(CVX_df)

# %% [markdown]
# # 4.2 - A markov chain model for the true price

# %%
from scipy.linalg import block_diag
import plotly.express as px

# %%
new_df = CVX_df.copy()
raw_spread = np.array(new_df["AskPrice"] - new_df["BidPrice"])
min_non_zero_spread = np.min(raw_spread[np.nonzero(raw_spread)])
min_non_zero_spread

# %%
rounded_spread = np.round(min_non_zero_spread, decimals = 2)
ticksize = rounded_spread
ticksize

# %%
new_df["Spread"] = np.round((new_df["AskPrice"] - new_df["BidPrice"])/ticksize)
new_df[["Spread"]]

# %%
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Histogram(x=new_df["Spread"]))
fig.show()

# %%
from scipy.stats import poisson
import plotly.graph_objects as go
lower_lim = 0
upper_lim = 25

lim_range = np.arange(0,25)
y = poisson.pmf(lim_range, 2.3) 
# Ensuring the area under the curves are (roughly) the same so they're on the same scale
scaled_y = y * np.sum(new_df["Spread"]) /1.7

line_plot_df = pd.DataFrame({"x":lim_range, "y":scaled_y})
fig = go.Figure()
fig.add_trace(go.Histogram(x=new_df["Spread"]))
fig.add_trace(go.Scatter(x=lim_range, y=scaled_y, line = dict(color = "red")))

# %%
small_new_df = new_df[(new_df["Spread"] <= 7) & (new_df["Spread"] != 0)].copy()

# %%
imb_bins = np.linspace(-1, 1, 101)
bin_midpoints = (imb_bins[1:] + imb_bins[:-1])/2
new_df["BinnedImbalance"] = pd.cut(new_df["Imbalance"], bins = imb_bins, labels = bin_midpoints)
fig = go.Figure()
fig.add_trace(go.Histogram(x=new_df["BinnedImbalance"], nbinsx = 101))

# %%
n_samples = 50000
lower_lim = 1
upper_lim = 100
bid_quantities = np.random.randint(lower_lim, upper_lim, size=n_samples)
ask_quantities = np.random.randint(lower_lim, upper_lim, size = n_samples)
imbalances = (bid_quantities - ask_quantities) / (bid_quantities + ask_quantities)
test_df = pd.DataFrame({"Imbalance":imbalances})
fig = go.Figure()
fig.add_trace(go.Histogram(x=test_df["Imbalance"],nbinsx=101))
fig.update_layout(title =f"Expected Imbalance if quantities are Unif[{lower_lim},{upper_lim}]")

# %%
# Poisson
# ------ 生成独立的泊松样本并画出 \bar I 的直方图（多种 λ 叠加） ------

# 固定随机种子，保证可复现（可选）
np.random.seed(42)

# 取不同的 λ（均值 = 方差 = λ）；λ 越大，\bar I 越容易集中在 0 附近
lambdas = [2, 5, 10, 20]
n_samples = 50_000  # 每个 λ 生成的样本数

fig = go.Figure()
for lam in lambdas:
    # 生成独立的买/卖数量（泊松分布）
    bid_q = np.random.poisson(lam=lam, size=n_samples)
    ask_q = np.random.poisson(lam=lam, size=n_samples)

    # 过滤掉分母为 0 的情形（bid_q=ask_q=0）
    mask = (bid_q + ask_q) > 0
    bid_q = bid_q[mask]
    ask_q = ask_q[mask]

    # 计算 \bar I = (Q^b - Q^a) / (Q^b + Q^a)
    imb = (bid_q - ask_q) / (bid_q + ask_q)

    # 叠加直方图；bins 与前文保持一致
    fig.add_trace(
        go.Histogram(
            x=imb,
            nbinsx=101,
            name=f"Poisson λ={lam}",
            opacity=0.55
        )
    )

# 设置为叠加显示，方便比较曲线形状
fig.update_layout(
    barmode="overlay",
    title="Imbalance (\\bar{I}) under independent Poisson counts (multiple λ)",
    xaxis_title="Imbalance ( (Q^b - Q^a) / (Q^b + Q^a) )",
    yaxis_title="Count"
)
fig.show()
# ------


# %%
# Geometric
# ------ 使用几何分布（右偏、重尾更明显）做对比，多种 p 叠加 ------

np.random.seed(43)  # 与上面区分开（可选）

# Numpy 的几何分布（geometric）取值从 1 开始，E[X] = 1/p
# p 越小，均值越大、尾部越重
p_list = [0.05, 0.1, 0.2, 0.3]
n_samples = 50_000

fig = go.Figure()
for p in p_list:
    # 生成独立的买/卖数量（几何分布，最小为 1 → 不会出现分母 0）
    bid_q = np.random.geometric(p, size=n_samples)
    ask_q = np.random.geometric(p, size=n_samples)

    # 直接计算 \bar I（几何分布不会出现 0，因此无需过滤）
    imb = (bid_q - ask_q) / (bid_q + ask_q)

    fig.add_trace(
        go.Histogram(
            x=imb,
            nbinsx=101,
            name=f"Geometric p={p}",
            opacity=0.55
        )
    )

fig.update_layout(
    barmode="overlay",
    title="Imbalance (\\bar{I}) under independent Geometric counts (multiple p)",
    xaxis_title="Imbalance ( (Q^b - Q^a) / (Q^b + Q^a) )",
    yaxis_title="Count"
)
fig.show()
# ------


# %%
small_new_df["Imbalance"] = small_new_df["BidQuantity"]/(small_new_df["BidQuantity"]+small_new_df["AskQuantity"])
imb_bins = np.linspace(0, 1, 51)
bin_midpoints = (imb_bins[1:] + imb_bins[:-1])/2
new_df["BinnedImbalance"] = pd.cut(new_df["Imbalance"], bins = imb_bins, labels = bin_midpoints)
fig = go.Figure()
fig.add_trace(go.Histogram(x=new_df["BinnedImbalance"], nbinsx = 101))

# %%
n_imbalances = 5
small_new_df["BinnedImbalance"] = pd.qcut(small_new_df["Imbalance"], n_imbalances, labels = False)
fig = go.Figure()
fig.add_trace(go.Histogram(x=small_new_df["BinnedImbalance"], nbinsx = 2*n_imbalances+1))
# Can see we have a uniform distribution now

# %%
# Function here
def discretize_values(dataframe, n_spreads=7, n_imbalances=5):
    """
    对 LOB 数据做离散化与筛选：
      1) 估计 tick size（最小价位跳动）；
      2) 计算 Spread（点差）并离散为“tick 的倍数”，记为整数列 'Spread'；
      3) 过滤掉 Spread==0 与 Spread>n_spreads 的行；
      4) 计算 Imbalance = Qb/(Qb+Qa)，并用分位数分箱到 0..n_imbalances-1；
    
    参数
    ----
    dataframe : pd.DataFrame
        需要包含列：'BidPrice','AskPrice','BidQuantity','AskQuantity'（以及 Datetime 不强制）
    n_spreads : int, default=7
        允许的最大点差倍数 m（即仅保留 Spread in {1,2,...,m} 的行）
    n_imbalances : int, default=5
        Imbalance 分箱数量（qcut 按分位数划分）

    返回
    ----
    small_df : pd.DataFrame
        过滤与离散化后的新 DataFrame（不修改原 DataFrame）
    """
    # --- 0) 拷贝一份，避免修改原数据 ---
    df = dataframe.copy()

    # --- 1) 估计 tick size：取最小非零的 Ask-Bid，然后做小数位四舍五入 ---
    raw_spread = (df["AskPrice"] - df["BidPrice"]).to_numpy()
    # 过滤出非零点差，避免 0 干扰
    nonzero = raw_spread[np.nonzero(raw_spread)]
    if nonzero.size == 0:
        raise ValueError("无法估计 tick size：数据中没有非零点差。")
    min_non_zero_spread = np.min(nonzero)
    # 为了稳妥处理 0.009999... 这类浮点误差，保留两位小数（本项目价位跳动设定在 0.01 量级）
    ticksize = np.round(min_non_zero_spread, decimals=2)

    # --- 2) 计算离散化的 Spread：四舍五入到“tick 倍数”的整数 ---
    # 先将原始点差除以 ticksize → 得到倍数，然后对倍数四舍五入取整
    spread_multiple = np.round((df["AskPrice"] - df["BidPrice"]) / ticksize).astype(int)
    df["Spread"] = spread_multiple  # 存成整数倍数（例如 1 表示 1*tick，2 表示 2*tick）

    # --- 3) 过滤：剔除 Spread==0（mid 可直接用，不参与马尔可夫）与 Spread>n_spreads 的样本 ---
    small_df = df[(df["Spread"] != 0) & (df["Spread"] <= n_spreads)].copy()

    # --- 4) 计算 Imbalance = Qb/(Qb+Qa)，并丢掉分母为 0 的行 ---
    denom = (small_df["BidQuantity"] + small_df["AskQuantity"]).to_numpy()
    valid = denom > 0
    small_df = small_df[valid].copy()
    small_df["Imbalance"] = small_df["BidQuantity"] / (small_df["BidQuantity"] + small_df["AskQuantity"])

    # --- 5) 用分位数分箱（qcut）把 Imbalance 离散化为 0..n_imbalances-1 ---
    # duplicates='drop' 防止重复分位点导致报错；若唯一值太少会自动降箱数
    small_df["BinnedImbalance"] = pd.qcut(
        small_df["Imbalance"],
        q=n_imbalances,
        labels=False,
        duplicates="drop"
    ).astype(int)

    # （可选）保留估计到的 ticksize，便于后续函数使用：
    # small_df.attrs["ticksize"] = ticksize

    return small_df


# %%
CVX_df["Imbalance"] = CVX_df["BidQuantity"]/(CVX_df["BidQuantity"]+CVX_df["AskQuantity"])
small_new_df== discretize_values(CVX_df)

# %%
def prepare_and_symmetrize(dataframe, timestep, n_imbalances = 5):
    df = dataframe.copy()
    df["NextMidPrice"] = df["MidPrice"].shift(-timestep)
    df["NextSpread"] = df["Spread"].shift(-timestep)
    df["NextDatetime"] = df["Datetime"].shift(-timestep)
    df["NextBinnedImbalance"] = df["BinnedImbalance"].shift(-timestep)
    # Rounding to the nearest integer, then dividing again
    # The mid price change takes values in [-0.01, -0.005,0, 0.005, 0.01] in our assumption
    # and we're rounding each of these here:
    df["MidChange"] = np.round(2*(df["NextMidPrice"] - df["MidPrice"])/ticksize)*ticksize/2
    # Removing the rows with mid price change not in [-0.01, -0.005,0, 0.005, 0.01].
    df = df[(df["MidChange"] <= ticksize * 1.1)&(df["MidChange"] >= -ticksize * 1.1)]

    # Symmetry:
    rev_df = df.copy(deep = True)
    rev_df["BinnedImbalance"] = n_imbalances - 1 - rev_df["BinnedImbalance"]
    rev_df["NextBinnedImbalance"] = n_imbalances - 1 - rev_df["NextBinnedImbalance"]
    rev_df["MidChange"] = -rev_df["MidChange"]
    rev_df["MidPrice"] = -rev_df["MidPrice"]
    return_df = pd.concat([df, rev_df])
    return_df["Index"] = pd.RangeIndex(return_df.shape[0])
    return_df.set_index("Index")
    return return_df

# %%
def fit_price_model(dataframe, n_spreads = 7, n_imbalances = 5):
    df = dataframe.copy()
    no_move=df[df["MidChange"]==0]
    no_move_counts=no_move.pivot_table(index=[ "NextBinnedImbalance"], 
                                        columns=["Spread", "BinnedImbalance"], 
                                        values="Datetime",
                                        fill_value=0, 
                                        aggfunc="count").unstack()
    # This gives us the first block of the Q matrix - for the positions where the mid price didn't change
    Q_counts=np.resize(np.array(no_move_counts[0:(n_imbalances*n_imbalances)]),(n_imbalances,n_imbalances))
    for i in range(1,n_spreads):
        # Reshaping the i'th block of the Q matrix
        Qi=np.resize(np.array(
                no_move_counts[(i*n_imbalances*n_imbalances):(i+1)*(n_imbalances*n_imbalances)]),
                        (n_imbalances,n_imbalances))
        # Putting this at the end of the Q matrix forming a block diagonal iteratively.
        Q_counts=block_diag(Q_counts,Qi)

    # Doing the same for positions where it moves:
    moves = df[df["MidChange"]!=0]
    move_counts = moves.pivot_table(index=["MidChange"], 
                            columns=["Spread", "BinnedImbalance"], 
                            values="Datetime",
                            fill_value=0, 
                            aggfunc="count").unstack()
    K = np.array([-0.01, -0.005, 0.005, 0.01])
    n_mid_changes = len(K)
    R_counts = np.resize(np.array(move_counts), (n_imbalances*n_spreads, n_mid_changes))
    T_v1 = np.concatenate((Q_counts, R_counts), axis = 1).astype(float)
    # Normalising to have sum 1.
    for i in range(0, n_imbalances*n_spreads):
        T_v1[i] = T_v1[i]/T_v1[i].sum()
    Q = T_v1[:,0:(n_imbalances*n_spreads)]
    R_1 = T_v1[:,(n_imbalances*n_spreads):]

    # Counts using different indices:
    move_counts = moves.pivot_table(index=["Spread", "BinnedImbalance"], 
                            columns=["NextSpread", "NextBinnedImbalance"], 
                            values="Datetime",
                            fill_value=0, 
                            aggfunc='count').unstack()

    R_2_counts = np.resize(np.array(move_counts), (n_imbalances * n_spreads, n_imbalances*n_spreads))
    T_v2 = np.concatenate((Q_counts, R_2_counts), axis=1).astype(float)

    # Normalising again
    for i in range(0, n_imbalances*n_spreads):
        T_v2[i] = T_v2[i] / T_v2[i].sum()
    R_2 = T_v2[:, (n_imbalances*n_spreads):]
    Q_2=T_v2[:,0:(n_imbalances*n_spreads)]
    G1 = np.dot(np.dot(np.linalg.inv(np.eye(n_imbalances*n_spreads)-Q), R_1),K)
    B = np.dot(np.linalg.inv(np.eye(n_imbalances*n_spreads)-Q),R_2)
    return(G1, B, Q, Q_2, R_1, R_2, K)

# %%
def get_test_train_dfs(dataframe, split_prop = 0.8):
    n_samples = dataframe.shape[0]
    split_point = int(np.round(n_samples * split_prop))
    train_df = dataframe.iloc[0:split_point].copy()
    print(split_point)
    test_df = dataframe.iloc[(split_point+1):n_samples].copy()
    return train_df, test_df
    
ticker = "CVX"
ticker_df =  get_sample_dataframe(ticker)
add_mid(ticker_df, inplace=  True)
ticker_df["Imbalance"] = ticker_df["BidQuantity"] / (ticker_df["BidQuantity"] + ticker_df["AskQuantity"])
train_df, test_df = get_test_train_dfs(ticker_df,split_prop = 0.4)

# Fitting the model on the train part:
n_imbalances = 5,
# Setting an appropriate number of spreads.
n_spreads = 7

if ticker == "BAC":
    n_spreads = 2 # fill in default your choice of n_spreads, n_imbalances here.
    n_imbalances = 5
else:
    n_spreads = 7
    n_imbalances = 5
disc_train = discretize_values(train_df, n_spreads=n_spreads, n_imbalances=n_imbalances)
symm_disc_train = prepare_and_symmetrize(disc_train, 1, n_imbalances=n_imbalances)
G1, B, Q, Q_2, R_1, R_2, K = fit_price_model(symm_disc_train, n_spreads = n_spreads, n_imbalances=n_imbalances)

# %%
def compute_nth_order_adjustment(G1,B,n, n_spreads=7):
    G_list = []
    if n <= 1:
        raise ValueError("Already have G1")
    G_list.append(G1)
    B_power = np.identity(B.shape[0])
    for k in range(n-1):
        B_power = np.dot(B_power, B)
        Gi = G_list[len(G_list)-1] + np.dot(B_power, G1)
        G_list.append(Gi)
    
    Gn = G_list[len(G_list)-1]
    return np.reshape(Gn, (n_spreads,5))

# %%
n = 6
Gn= compute_nth_order_adjustment(G1,B,n, n_spreads = n_spreads)

spread = 2
imbalance = 3
spread_idx = spread-1
imbalance_idx = imbalance-1
# the adjustment from the midpoint for a 
# case with spread 2, and imbalance in bin 3
Gn[spread_idx, imbalance_idx] 

# %%
def get_adjustment(dataframe_row,matrix_G_star):
    spread_idx = dataframe_row["Spread"]-1
    imbalance_idx = dataframe_row["BinnedImbalance"]-1

    adjustment = matrix_G_star[spread_idx, imbalance_idx] 
    return adjustment

def estimate_ticksize(df, tick_decimals: int = 2):
    """
    在给定数据上估计最小价位跳动 tick size：
    取最小的非零 (Ask - Bid) 并四舍五入到 tick_decimals 位小数。
    """
    raw = (df["AskPrice"] - df["BidPrice"]).to_numpy()
    nonzero = raw[np.nonzero(raw)]
    if nonzero.size == 0:
        raise ValueError("无法估计 tick size：数据中没有非零点差。")
    # 处理 0.009999... 这类浮点误差：保留若干小数位
    tick = np.round(nonzero.min(), decimals=tick_decimals)
    return tick


def get_imbalance_bin_edges(train_df: pd.DataFrame, n_imbalances: int):
    """
    在训练集上按“分位数”确定失衡度分箱边界（用于 test 保持一致）。
    返回边界数组 edges（长度 n_imbalances+1，严格递增，落在 [0,1]）。
    """
    # 训练集需有 Imbalance = Qb/(Qb+Qa)
    imb = train_df["Imbalance"].dropna().to_numpy()
    if imb.size == 0:
        raise ValueError("训练集没有可用的 Imbalance。")
    # 取 0..1 等距分位点
    q = np.linspace(0, 1, n_imbalances + 1)
    edges = np.quantile(imb, q).astype(float)

    # 确保严格单调（离散数据时分位点可能重复），用一个极小 eps 处理重复边界
    eps = 1e-12
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + eps

    # 限幅到 [0,1]
    edges[0] = max(0.0, edges[0])
    edges[-1] = min(1.0, edges[-1])
    return edges


def compute_adjusted_price(
    dataframe: pd.DataFrame,
    Gn: np.ndarray,
    n_spreads: int = 7,
    n_imbalances: int = 5,
    train_df: pd.DataFrame = None,          # 新增：传入训练集以提取离散化规则
    tick_decimals: int = 2
):
    """
    用给定的 G*（n 阶调整矩阵）在 dataframe 上计算 MicroPrice。
    关键修复：若给了 train_df，就用训练集的 tick 与失衡分箱边界来“统一离散化”。

    参数
    ----
    dataframe : 待预测的数据（一般是 test_df）
    Gn        : n 阶累计调整矩阵，形状 (n_spreads, n_imbalances)
    n_spreads : 训练时使用的最大点差倍数 m
    n_imbalances : 训练时使用的失衡分箱数
    train_df  : 训练集（已含 Imbalance 列），用于提取 tick 与分箱边界
    tick_decimals : 估计 tick 时四舍五入的小数位

    返回
    ----
    result_df : 原数据 + 一列 'MicroPrice'
    """
    # 复制，保证不改原表
    result_df = dataframe.copy()

    # -----------------------------
    # 1) 统一离散化规则（来自训练集）
    # -----------------------------
    if train_df is None:
        # 没有训练集时，退化为“旧逻辑”，但会警告可能不一致
        # —— 如需强一致性，请务必传入 train_df
        # 这里沿用 test 自己的离散化（不推荐），以保持兼容
        disc_df = discretize_values(result_df, n_spreads=n_spreads, n_imbalances=n_imbalances)
        # 取出离散后的 Spread 与 BinnedImbalance
        spread_mult = disc_df["Spread"].astype(int).to_numpy()
        imb_bin = disc_df["BinnedImbalance"].astype(int).to_numpy()
        mid = disc_df["MidPrice"].to_numpy()
        idx = disc_df.index.to_numpy()
    else:
        # （推荐路径）基于“训练集规则”离散 test：
        # a) 训练集 tick
        tick = estimate_ticksize(train_df, tick_decimals=tick_decimals)
        # b) 测试集 Spread 的“tick 倍数整数”
        spread_mult = np.round((result_df["AskPrice"] - result_df["BidPrice"]) / tick).astype(int)
        # c) 训练集的失衡分箱边界
        edges = get_imbalance_bin_edges(train_df, n_imbalances=n_imbalances)
        # d) 测试集 Imbalance
        denom = (result_df["BidQuantity"] + result_df["AskQuantity"]).to_numpy()
        with np.errstate(divide='ignore', invalid='ignore'):
            imb = np.where(denom > 0, result_df["BidQuantity"].to_numpy() / denom, np.nan)
        # e) 用训练分位数边界在 test 上 cut 出 0..n_imbalances-1
        imb_bin = pd.cut(imb, bins=edges, labels=False, include_lowest=True)
        imb_bin = imb_bin.astype("float")  # 可能有 NaN
        mid = result_df["MidPrice"].to_numpy()
        idx = result_df.index.to_numpy()

    # -----------------------------
    # 2) 只对“可处理”的样本做调整：Spread in [1..n_spreads] 且 有效的失衡 bin
    # -----------------------------
    valid_spread = (spread_mult >= 1) & (spread_mult <= n_spreads)
    valid_imb = (~pd.isna(imb_bin)) & (imb_bin >= 0) & (imb_bin < n_imbalances)
    valid = valid_spread & valid_imb

    # 预设调整量为 0（无效的样本就回退到 mid）
    adjustment = np.zeros(len(result_df), dtype=float)

    # 把有效样本映射到 Gn 的行/列：注意 Python 索引从 0 开始
    s_idx = (spread_mult[valid].astype(int) - 1)
    i_idx = (imb_bin[valid].astype(int))
    adjustment[valid] = Gn[s_idx, i_idx]

    # -----------------------------
    # 3) 计算最终 MicroPrice：Mid + 调整；无效样本自动回退为 Mid
    # -----------------------------
    result_df["MicroPrice"] = mid + adjustment

    return result_df


pred_test = compute_adjusted_price(test_df, Gn)


# %%
pred_test.isna().sum()

# %%
add_mid(pred_test, inplace = True)
add_weighted_mid(pred_test, inplace = True)
add_NLWMPrice(pred_test, inplace = True)

plot_price_change(pred_test)

# %%



