
## **写在前面：竞赛核心思想**

在背板子之前，请牢记以下几点，它们比任何一个板子都重要：

1.  **审题是关键**：理解题意、数据范围、时间/空间限制，是决定算法方向的根本。
2.  **暴力出奇迹**：拿到题先想暴力解法。暴力解法不仅能帮你理解问题，很多时候还是正解的对拍工具。
3.  **由简到繁**：从部分分、小数据范围入手，逐步优化到正解。
4.  **复杂度意识**：时刻关注你的算法时间复杂度（通常是 `O(n^k)` 或 `O(n log n)` 等形式）和空间复杂度，确保能在规定时间内运行。
5.  **模板化思维**：将经典算法模块化，思考“这个问题能用哪个/哪些模板组合来解决？”
6.  **代码规范**：变量名清晰、适当注释、结构清晰。竞赛时，清晰易懂的代码是你调试的唯一救星。

---

### **一、 基础与工具**

#### **1. STL (Standard Template Library)**

STL是C++的精髓，竞赛中必须熟练使用。

*   **`vector`**：动态数组。
    *   **适用场景**：数组长度不确定，需要频繁在尾部增删。
    *   **常用操作**：`push_back()`, `pop_back()`, `size()`, `empty()`, `clear()`, `resize()`。
*   **`string`**：字符串。
    *   **适用场景**：所有字符串操作，比`char[]`方便安全。
    *   **常用操作**：`+=`, `length()`, `substr()`, `find()`, `c_str()` (用于C函数接口)。
*   **`queue`**：队列。
    *   **适用场景**：BFS，模拟宽度优先遍历过程。
    *   **常用操作**：`push()`, `pop()`, `front()`, `back()`, `empty()`。
*   **`priority_queue`**：优先队列（堆）。
    *   **适用场景**：动态求最大/最小值，Dijkstra算法。
    *   **板子**：`priority_queue<int> pq;` (大根堆)，`priority_queue<int, vector<int>, greater<int>> pq;` (小根堆)。
*   **`stack`**：栈。
    *   **适用场景**：DFS，表达式求值，模拟函数调用。
    *   **常用操作**：`push()`, `pop()`, `top()`, `empty()`。
*   **`deque`**：双端队列。
    *   **适用场景**：需要频繁在头尾操作的数据结构，如单调队列。
    *   **常用操作**：`push_front()`, `push_back()`, `pop_front()`, `pop_back()`。
*   **`set` / `multiset`**：集合（自动排序，唯一/不唯一）。
    *   **适用场景**：动态维护一个有序集合，快速查找、插入、删除。
    *   **常用操作**：`insert()`, `erase()`, `find()`, `count()`, `lower_bound()` (返回第一个 >= x 的迭代器), `upper_bound()` (返回第一个 > x 的迭代器)。
*   **`map` / `unordered_map`**：映射（键值对）。
    *   **`map`**：红黑树实现，`O(log n)`，按键有序。适用于需要有序遍历键的场景。
    *   **`unordered_map`**：哈希表实现，`O(1)`平均，按键无序。适用于只查不改，追求极致速度的场景。
    *   **常用操作**：`[key] = value`，`find(key)`，`count(key)`，`erase(key)`。
*   **常用算法**：`#include <algorithm>`
    *   `sort(a, a + n)`：排序。
    *   `reverse(a, a + n)`：反转。
    *   `unique(a, a + n)`：去重（需先排序）。返回去重后尾地址。
    *   `lower_bound(a, a + n, x)` / `upper_bound()`：二分查找。
    *   `max()`, `min()`, `abs()`, `swap()`。

#### **2. 高精度**

```cpp
// 高精度结构体
struct BigInt {
    vector<int> A;

    // 构造函数：从long long初始化
    BigInt(long long n = 0) {
        if (n == 0) A.push_back(0);
        while (n > 0) {
            A.push_back(n % 10);
            n /= 10;
        }
    }

    // 构造函数：从string初始化
    BigInt(string s) {
        for (int i = s.length() - 1; i >= 0; i--) {
            A.push_back(s[i] - '0');
        }
        trim();
    }

    // 去除前导零
    void trim() {
        while (A.size() > 1 && A.back() == 0) A.pop_back();
    }

    // 输出
    void print() {
        if (A.empty()) { cout << 0; return; }
        for (int i = A.size() - 1; i >= 0; i--) cout << A[i];
    }

    // 比较大小：小于
    bool operator < (const BigInt& b) const {
        if (A.size() != b.A.size()) return A.size() < b.A.size();
        for (int i = A.size() - 1; i >= 0; i--) {
            if (A[i] != b.A[i]) return A[i] < b.A[i];
        }
        return false; // 相等或大于
    }
    
    // 比较大小：大于
    bool operator > (const BigInt& b) const { return b < *this; }
    // 比较大小：小于等于
    bool operator <= (const BigInt& b) const { return !(*this > b); }
    // 比较大小：大于等于
    bool operator >= (const BigInt& b) const { return !(*this < b); }
    // 比较大小：相等
    bool operator == (const BigInt& b) const { return !(*this < b) && !(b < *this); }

    // --- 高精 + 高精 ---
    BigInt operator + (const BigInt& b) {
        BigInt C; C.A.clear();
        int t = 0;
        for (int i = 0; i < A.size() || i < b.A.size(); i++) {
            if (i < A.size()) t += A[i];
            if (i < b.A.size()) t += b.A[i];
            C.A.push_back(t % 10);
            t /= 10;
        }
        if (t) C.A.push_back(t);
        return C;
    }

    // --- 高精 - 高精 --- (需保证结果非负，否则逻辑需调整)
    BigInt operator - (const BigInt& b) {
        BigInt C = *this; // 拷贝一份
        // 如果 A < B，通常在外部处理负号并交换，这里只处理 A >= B
        // 简单处理：如果本对象小于b，返回0或报错，或者外部自行判断
        int t = 0;
        for (int i = 0; i < C.A.size(); i++) {
            t = C.A[i] - t;
            if (i < b.A.size()) t -= b.A[i];
            C.A[i] = (t + 10) % 10;
            if (t < 0) t = 1; else t = 0;
        }
        C.trim();
        return C;
    }

    // --- 高精 * 高精 ---
    BigInt operator * (const BigInt& b) {
        BigInt C;
        C.A.assign(A.size() + b.A.size(), 0);
        for (int i = 0; i < A.size(); i++) {
            for (int j = 0; j < b.A.size(); j++) {
                C.A[i + j] += A[i] * b.A[j];
            }
        }
        int t = 0;
        for (int i = 0; i < C.A.size(); i++) {
            C.A[i] += t;
            t = C.A[i] / 10;
            C.A[i] %= 10;
        }
        while (t) { C.A.push_back(t % 10); t /= 10; }
        C.trim();
        return C;
    }

    // --- 高精 / 低精 (int) ---
    // 返回 {商, 余数}
    pair<BigInt, int> div_mod(int b) {
        BigInt C; 
        int r = 0;
        for (int i = A.size() - 1; i >= 0; i--) {
            r = r * 10 + A[i];
            C.A.push_back(r / b);
            r %= b;
        }
        reverse(C.A.begin(), C.A.end()); // 因为是从高位算的，push进去是反的，要翻转回来
        C.trim();
        return {C, r};
    }
    
    // 重载除法运算符 (只返回商)
    BigInt operator / (int b) {
        return div_mod(b).first;
    }
    
    // 重载取模运算符 (只返回余数)
    int operator % (int b) {
        return div_mod(b).second;
    }
};

// --- 使用示例 ---
int main() {
    string s1, s2;
    // 输入两个大整数
    cin >> s1 >> s2;
    
    BigInt a(s1), b(s2);

    // 加法
    BigInt add = a + b;
    cout << "Add: "; add.print(); cout << endl;

    // 减法 (需要判断大小处理负号)
    cout << "Sub: ";
    if (a < b) {
        cout << "-";
        (b - a).print();
    } else {
        (a - b).print();
    }
    cout << endl;

    // 乘法
    BigInt mul = a * b;
    cout << "Mul: "; mul.print(); cout << endl;

    // 除法 (高精除以低精，这里假设 s2 能装进 int，如果 s2 也是超大数，除法逻辑会非常复杂)
    // 演示：a / 123
    int divisor = 123;
    pair<BigInt, int> res = a.div_mod(divisor);
    cout << "Div (a / " << divisor << "): "; res.first.print(); cout << endl;
    cout << "Mod (a % " << divisor << "): " << res.second << endl;

    return 0;
}
```
---

### **二、 基础算法**

#### **1. 排序**

```cpp
struct Node {
    int id;
    int score;
    int time;
};

// 比较规则：
// 1. 分数高的排前面
// 2. 分数一样，用时少的排前面
bool cmp(const Node& a, const Node& b) {
    if (a.score != b.score) return a.score > b.score; // 降序
    return a.time < b.time; // 升序
}

int main() {
    vector<int> a = {3, 1, 4, 1, 5, 9};
    
    // 1. 基础排序（默认升序）
    sort(a.begin(), a.end()); 
    
    // 2. 降序 (使用 greater<int>())
    sort(a.begin(), a.end(), greater<int>());

    // 3. 结构体排序
    vector<Node> nodes = {{1, 100, 20}, {2, 90, 10}, {3, 100, 15}};
    sort(nodes.begin(), nodes.end(), cmp);
    
    // 4. Lambda 写法 (最常用，省去写cmp函数的麻烦)
    sort(nodes.begin(), nodes.end(), [](const Node& x, const Node& y) {
        if (x.score != y.score) return x.score > y.score;
        return x.time < y.time;
    });

    return 0;
}
```

```c++
// 求逆序对采用
const int N = 100005;
int q[N], tmp[N]; // tmp用于归并时的辅助存储
long long ans = 0; // 记录逆序对数量 (注意用long long)

void merge_sort(int l, int r) {
    if (l >= r) return;

    int mid = l + r >> 1; // 相当于 (l+r)/2
    merge_sort(l, mid);
    merge_sort(mid + 1, r);

    // 归并过程
    int k = 0, i = l, j = mid + 1;
    while (i <= mid && j <= r) {
        if (q[i] <= q[j]) {
            tmp[k++] = q[i++];
        } else {
            tmp[k++] = q[j++];
            // 核心代码：统计逆序对
            // 如果 q[i] > q[j]，说明 q[i...mid] 都大于 q[j]
            ans += (mid - i + 1); 
        }
    }
    
    // 扫尾
    while (i <= mid) tmp[k++] = q[i++];
    while (j <= r) tmp[k++] = q[j++];

    // 复制回原数组
    for (int i = l, j = 0; i <= r; i++, j++) q[i] = tmp[j];
}

int main() {
    int n;
    cin >> n;
    for(int i = 0; i < n; i++) cin >> q[i];
    
    merge_sort(0, n - 1);
    
    cout << ans << endl; // 输出逆序对数量
    return 0;
}
```

```
// 找第k大的数
int main() {
    vector<int> a = {5, 2, 4, 1, 3};
    int k = 2; // 找第 3 小的数 (下标从0开始，k=2即第3个)

    // nth_element 会保证 a[k] 是正确排好序后的元素
    // 且 a[0...k-1] 都 <= a[k]，a[k+1...n] 都 >= a[k]
    // 复杂度 O(N)
    nth_element(a.begin(), a.begin() + k, a.end());
    
    cout << a[k]; // 输出 3
}
// 或者手写
int quick_select(int l, int r, int k) {
    if (l == r) return q[l];

    int i = l - 1, j = r + 1, x = q[l + r >> 1];
    while (i < j) {
        do i++; while (q[i] < x);
        do j--; while (q[j] > x);
        if (i < j) swap(q[i], q[j]);
    }

    // 此时数组被 j 分割为左半边长度 sl
    int sl = j - l + 1;
    // 如果 k 在左半边
    if (k <= sl) return quick_select(l, j, k);
    // 如果 k 在右半边，找右半边的第 k - sl 个
    else return quick_select(j + 1, r, k - sl);
}
```

```C++
// 快排模版
const int N = 100005;
int q[N];

void quick_sort(int l, int r) {
    if (l >= r) return;

    // 重点：pivot 取中间值，不要取 q[l] 或 q[r]，防止退化
    int i = l - 1, j = r + 1, x = q[l + r >> 1];
    
    while (i < j) {
        do i++; while (q[i] < x);
        do j--; while (q[j] > x);
        if (i < j) swap(q[i], q[j]);
    }
    
    // 递归处理
    quick_sort(l, j);
    quick_sort(j + 1, r);
}

int main() {
    int n;
    cin >> n;
    for(int i = 0; i < n; i++) cin >> q[i];
    quick_sort(0, n - 1);
    for(int i = 0; i < n; i++) cout << q[i] << " ";
    return 0;
}
```

#### **2.记忆化搜索**

*   **核心思想**：带备忘录的递归。在递归过程中，将计算过的结果存下来（通常用数组或哈希表），下次遇到相同子问题时直接返回结果。
*   **适用场景**：DFS树/图中有大量重复子问题，本质是自顶向下的动态规划。
*   **板子思路**：
    ```cpp
    int memo[N]; // 初始化为一个特殊值，如-1，表示未计算
    int solve(int x) {
        if (x == /*基例*/) return /*基例答案*/;
        if (memo[x] != -1) return memo[x];
        int res = /*根据递归式计算*/;
        return memo[x] = res;
    }
    ```

#### **3. 前缀和与差分**

*   **前缀和**：
    *   **适用场景**：快速查询数组 `[l, r]` 区间内元素的和。`O(1)` 查询。
    *   **板子**：`sum[i] = sum[i-1] + a[i]`。查询 `sum(r) - sum(l-1)`。
*   **差分**：
    *   **适用场景**：快速对数组 `[l, r]` 区间内所有元素加上一个值 `c`。`O(1)` 修改。
    *   **思想**：差分数组 `d` 的前缀和是原数组 `a`。
    *   **板子**：对 `a` 的 `[l, r]` 加 `c`，操作为：`d[l] += c`, `d[r+1] -= c`。最后对 `d` 求一遍前缀和即可得到修改后的 `a`。
    *   **应用**：二维差分、区间修改类问题。

---

### **三、 搜索**

#### **1. DFS (Depth-First Search)**

*   **核心思想**：一条路走到黑，走不通再回头换路。用**栈**（递归栈）实现。
*   **适用场景**：
    *   图和树的遍历。
    *   求解所有可能性（排列组合、子集）。
    *   拓扑排序、染色法判二分图。
    *   走迷宫（但求最短路不如BFS）。
*   **板子 (递归)**：

```cpp
vector<int> adj[N]; // 邻接表存图
bool visited[N];

void dfs(int u) {
    visited[u] = true;
    // 处理顶点 u
    for (int v : adj[u]) {
        if (!visited[v]) {
            dfs(v);
        }
    }
}
```

#### **2. BFS (Breadth-First Search)**

*   **核心思想**：一层一层地向外扩展。用**队列**实现。
*   **适用场景**：
    *   **无权图的最短路问题**。保证第一次访问到终点时就是最短路径。
    *   拓扑排序。
    *   在状态转移图中求解最少步数问题。
*   **板子**：

```cpp
#include <queue>
vector<int> adj[N];
int dist[N]; // 存储距离

void bfs(int start_node) {
    queue<int> q;
    memset(dist, -1, sizeof(dist)); // -1 表示未访问
    q.push(start_node);
    dist[start_node] = 0;

    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int v : adj[u]) {
            if (dist[v] == -1) { // 未访问
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        }
    }
}
```

---

### **四、 字符串**

#### **1. KMP (Knuth-Morris-Pratt)**

*   **适用场景**：在一个长字符串 `S` 中查找一个短字符串 `P` 是否出现及出现位置。
*   **核心思想**：利用匹配失败后的信息，减少模式串与主串的匹配次数。
    *   **`next` 数组**：`next[i]` 表示 `P` 的子串 `P[0...i]` 的最长公共前后缀长度。当 `P[i]` 匹配失败时，模式串跳转到 `next[i-1]` 的位置继续匹配。
*   **板子**：

```cpp
vector<int> get_next(const string& p) {
    int n = p.length();
    vector<int> next(n, 0);
    for (int i = 1, j = 0; i < n; ++i) {
        while (j > 0 && p[i] != p[j]) {
            j = next[j - 1];
        }
        if (p[i] == p[j]) {
            j++;
        }
        next[i] = j;
    }
    return next;
}

int kmp_search(const string& s, const string& p) {
    int n = s.length(), m = p.length();
    vector<int> next = get_next(p);
    for (int i = 0, j = 0; i < n; ++i) {
        while (j > 0 && s[i] != p[j]) {
            j = next[j - 1];
        }
        if (s[i] == p[j]) {
            j++;
        }
        if (j == m) { // 找到一个匹配
            return i - m + 1; // 返回起始位置
        }
    }
    return -1; // 未找到
}
```

#### **2. 字符串哈希**

*   **适用场景**：快速判断两个字符串是否相等，尤其在解决字符串匹配问题时。
*   **核心思想**：将字符串看作一个**B进制的数**，对一个大的质数 `M` 取余，得到一个哈希值。
    *   `H = (s[0]*B^(n-1) + s[1]*B^(n-2) + ... + s[n-1]*B^0) mod M`
*   **板子 (前缀哈希)**：

```cpp
const unsigned long long B = 137; // 进制，常用小质数
unsigned long long h[N], p_pow[N]; // h[i]前缀哈希, p_pow[i]B的i次方

void init_hash(const string& s) {
    int n = s.length();
    p_pow[0] = 1;
    h[0] = s[0];
    for (int i = 1; i < n; ++i) {
        h[i] = h[i - 1] * B + (unsigned long long)s[i];
        p_pow[i] = p_pow[i - 1] * B;
    }
}

// 获取 s[l...r] 的哈希值
unsigned long long get_hash(int l, int r) {
    if (l == 0) return h[r];
    return h[r] - h[l - 1] * p_pow[r - l + 1];
}
```
*   **注意事项**：哈希冲突。可以采用双哈希（用两个不同的B和M）来降低冲突概率。在 `unsigned long long` 下自然溢出（相当于对 `2^64` 取模）通常足够快且冲突概率低。

---

### **五、 数论**

#### **1. 最大公约数 与最小公倍数**

*   **板子 (欧几里得算法)**：

```cpp
long long gcd(long long a, long long b) {
    return b == 0 ? a : gcd(b, a % b);
}
long long lcm(long long a, long long b) {
    return a / gcd(a, b) * b; // 先除后乘，防止溢出
}
```

#### **2. 扩展欧几里得**

*   **适用场景**：求解 `ax + by = gcd(a, b)` 的一组整数解 `(x, y)`。可用于求解模逆元。
*   **板子**：

```cpp
long long exgcd(long long a, long long b, long long& x, long long& y) {
    if (b == 0) {
        x = 1;
        y = 0;
        return a;
    }
    long long gcd_val = exgcd(b, a % b, y, x);
    y -= (a / b) * x;
    return gcd_val;
}
```

#### **3. 快速幂**

*   **适用场景**：快速计算 `a^b mod m`。
*   **核心思想**：二分，`a^b = a^(b/2) * a^(b/2)` (b为偶) 或 `a^b = a * a^(b/2) * a^(b/2)` (b为奇)。
*   **板子**：

```cpp
long long qpow(long long a, long long b, long long m) {
    long long res = 1 % m;
    a %= m;
    while (b > 0) {
        if (b & 1) res = res * a % m;
        a = a * a % m;
        b >>= 1;
    }
    return res;
}
```

#### **4. 素数筛**

*   **适用场景**：获取一定范围内的所有素数。
*   **埃氏筛**：`O(n log log n)`。从2开始，把每个素数的倍数都标记为合数。
*   **欧拉筛**：`O(n)`。每个合数只被其最小的质因数筛掉一次，效率更高。
*   **板子 (欧拉筛)**：

```cpp
vector<int> primes;
bool is_composite[N];

void linear_sieve(int n) {
    for (int i = 2; i <= n; ++i) {
        if (!is_composite[i]) {
            primes.push_back(i);
        }
        for (int p : primes) {
            if (p * i > n) break;
            is_composite[p * i] = true;
            if (i % p == 0) break; // 关键步骤
        }
    }
}
```

#### **5. 逆元**

*   **适用场景**：在有模数的情况下，计算除法 `(a/b) mod m`，等价于 `a * b^{-1} mod m`。
*   **求法**：
    1.  当 `m` 为质数时，`b^{-1} = b^(m-2) mod m` (费马小定理 + 快速幂)。
    2.  当 `gcd(b, m) = 1` 时，用扩展欧几里得解 `bx + my = 1`，`x` 即为逆元。

---



### **六、 图论**

#### **1. 图的存储**

*   **邻接矩阵**：`O(n^2)` 空间，适合稠密图，判断两点是否有边快 `O(1)`。
*   **邻接表**：`O(n+m)` 空间，适合稀疏图，遍历一个点的所有邻居 `O(deg(v))`。竞赛常用。
    ```cpp
    vector<pair<int, int>> adj[N]; // adj[u].push_back({v, w})
    ```

#### **2. 并查集**

*   **适用场景**：处理动态连通性问题，如合并集合、查询两个元素是否在同一集合。
*   **核心思想**：用树形结构表示集合，查找祖先、合并集合。
*   **优化**：**路径压缩** 和 **按秩合并**，使得操作均摊复杂度接近 `O(1)`。
*   **板子**：

```cpp
int parent[N];
int rank[N]; // 或者用 size[N]

void init(int n) {
    for (int i = 0; i <= n; ++i) {
        parent[i] = i;
        rank[i] = 0;
    }
}

int find(int u) {
    if (parent[u] != u) {
        parent[u] = find(parent[u]); // 路径压缩
    }
    return parent[u];
}

void unite(int u, int v) {
    u_root = find(u);
    v_root = find(v);
    if (u_root == v_root) return;

    // 按秩合并
    if (rank[u_root] < rank[v_root]) {
        parent[u_root] = v_root;
    } else {
        parent[v_root] = u_root;
        if (rank[u_root] == rank[v_root]) {
            rank[u_root]++;
        }
    }
}
```

#### **3. 最短路算法**

*   **Dijkstra**：
    *   **适用场景**：**单源**最短路，**边权为正**。
    *   **板子 (堆优化)**：
    
    ```cpp
    class Dijkstra {
    public:
        // 邻接表版本
        vector<int> shortestPath(int n, vector<vector<pair<int, int>>>& graph, int start) {
            vector<int> dist(n, INT_MAX);
            dist[start] = 0;
            
            // 最小堆：pair<距离, 节点>
            priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
            pq.push({0, start});
            
            while (!pq.empty()) {
                auto [currentDist, u] = pq.top();
                pq.pop();
                
                // 如果当前距离大于已知最短距离，跳过
                if (currentDist > dist[u]) continue;
                
                for (auto& [v, weight] : graph[u]) {
                    int newDist = currentDist + weight;
                    if (newDist < dist[v]) {
                        dist[v] = newDist;
                        pq.push({newDist, v});
                    }
                }
            }
            
            return dist;
        }
        
        // 邻接矩阵版本
        vector<int> shortestPathMatrix(int n, vector<vector<int>>& graph, int start) {
            vector<int> dist(n, INT_MAX);
            vector<bool> visited(n, false);
            dist[start] = 0;
            
            for (int i = 0; i < n; i++) {
                // 找到未访问的最小距离节点
                int u = -1;
                int minDist = INT_MAX;
                for (int j = 0; j < n; j++) {
                    if (!visited[j] && dist[j] < minDist) {
                        minDist = dist[j];
                        u = j;
                    }
                }
                
                if (u == -1) break;  // 所有节点都已访问
                
                visited[u] = true;
                
                // 更新邻接节点距离
                for (int v = 0; v < n; v++) {
                    if (!visited[v] && graph[u][v] != INT_MAX) {
                        dist[v] = min(dist[v], dist[u] + graph[u][v]);
                    }
                }
            }
            
            return dist;
        }
        
        // 获取最短路径（需要记录前驱）
        vector<int> getPath(int n, vector<vector<pair<int, int>>>& graph, int start, int end) {
            vector<int> dist(n, INT_MAX);
            vector<int> prev(n, -1);
            dist[start] = 0;
            
            priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
            pq.push({0, start});
            
            while (!pq.empty()) {
                auto [currentDist, u] = pq.top();
                pq.pop();
                
                if (currentDist > dist[u]) continue;
                
                for (auto& [v, weight] : graph[u]) {
                    int newDist = currentDist + weight;
                    if (newDist < dist[v]) {
                        dist[v] = newDist;
                        prev[v] = u;
                        pq.push({newDist, v});
                    }
                }
            }
            
            // 重构路径
            if (dist[end] == INT_MAX) return {};
            
            vector<int> path;
            for (int at = end; at != -1; at = prev[at]) {
                path.push_back(at);
            }
            reverse(path.begin(), path.end());
            return path;
        }
    };
    ```
    
*   **Bellman-Ford**：
    * **适用场景**：**单源**最短路，**边权可负**，可用于**判断负权环**。
    
      ```c++
      class BellmanFord {
      public:
          // 返回dist数组，如果存在负环则返回空数组
          vector<int> shortestPath(int n, vector<vector<int>>& edges, int start) {
              vector<int> dist(n, INT_MAX);
              dist[start] = 0;
              
              // 松弛n-1次
              for (int i = 0; i < n - 1; i++) {
                  bool updated = false;
                  for (auto& edge : edges) {
                      int u = edge[0], v = edge[1], w = edge[2];
                      if (dist[u] != INT_MAX && dist[u] + w < dist[v]) {
                          dist[v] = dist[u] + w;
                          updated = true;
                      }
                  }
                  if (!updated) break;  // 提前终止
              }
              
              // 检查负环
              for (auto& edge : edges) {
                  int u = edge[0], v = edge[1], w = edge[2];
                  if (dist[u] != INT_MAX && dist[u] + w < dist[v]) {
                      return {};  // 存在负环
                  }
              }
              
              return dist;
          }
          
          // 检测负环
          bool hasNegativeCycle(int n, vector<vector<int>>& edges) {
              vector<int> dist(n, 0);  // 初始化为0，可以检测所有负环
              
              for (int i = 0; i < n; i++) {
                  for (auto& edge : edges) {
                      int u = edge[0], v = edge[1], w = edge[2];
                      if (dist[u] + w < dist[v]) {
                          dist[v] = dist[u] + w;
                          if (i == n - 1) return true;  // 第n次仍然能更新，存在负环
                      }
                  }
              }
              
              return false;
          }
          
          // 获取前驱和路径
          vector<int> getPath(int n, vector<vector<int>>& edges, int start, int end) {
              vector<int> dist(n, INT_MAX);
              vector<int> prev(n, -1);
              dist[start] = 0;
              
              // 松弛n-1次
              for (int i = 0; i < n - 1; i++) {
                  for (auto& edge : edges) {
                      int u = edge[0], v = edge[1], w = edge[2];
                      if (dist[u] != INT_MAX && dist[u] + w < dist[v]) {
                          dist[v] = dist[u] + w;
                          prev[v] = u;
                      }
                  }
              }
              
              // 检查负环
              for (auto& edge : edges) {
                  int u = edge[0], v = edge[1], w = edge[2];
                  if (dist[u] != INT_MAX && dist[u] + w < dist[v]) {
                      return {};  // 存在负环
                  }
              }
              
              // 重构路径
              if (dist[end] == INT_MAX) return {};
              
              vector<int> path;
              for (int at = end; at != -1; at = prev[at]) {
                  path.push_back(at);
              }
              reverse(path.begin(), path.end());
              return path;
          }
      };
      ```
    
*   **SPFA (Shortest Path Faster Algorithm)**：
    
    * **适用场景**：Bellman-Ford的队列优化版，实践中速度快。但容易被卡到最坏复杂度。
    
      ```c++
      class SPFA {
      public:
          // 返回dist数组，如果存在负环则返回空数组
          vector<int> shortestPath(int n, vector<vector<pair<int, int>>>& graph, int start) {
              vector<int> dist(n, INT_MAX);
              vector<bool> inQueue(n, false);
              vector<int> count(n, 0);  // 记录入队次数，用于检测负环
              
              dist[start] = 0;
              queue<int> q;
              q.push(start);
              inQueue[start] = true;
              count[start]++;
              
              while (!q.empty()) {
                  int u = q.front();
                  q.pop();
                  inQueue[u] = false;
                  
                  for (auto& [v, weight] : graph[u]) {
                      if (dist[u] != INT_MAX && dist[u] + weight < dist[v]) {
                          dist[v] = dist[u] + weight;
                          
                          if (!inQueue[v]) {
                              q.push(v);
                              inQueue[v] = true;
                              count[v]++;
                              
                              // 如果一个节点入队次数超过n次，存在负环
                              if (count[v] > n) {
                                  return {};  // 存在负环
                              }
                          }
                      }
                  }
              }
              
              return dist;
          }
          
          // 检测负环
          bool hasNegativeCycle(int n, vector<vector<pair<int, int>>>& graph) {
              vector<int> dist(n, 0);
              vector<bool> inQueue(n, true);  // 初始所有节点都在队列中
              vector<int> count(n, 0);
              
              queue<int> q;
              for (int i = 0; i < n; i++) {
                  q.push(i);
              }
              
              while (!q.empty()) {
                  int u = q.front();
                  q.pop();
                  inQueue[u] = false;
                  
                  for (auto& [v, weight] : graph[u]) {
                      if (dist[u] + weight < dist[v]) {
                          dist[v] = dist[u] + weight;
                          
                          if (!inQueue[v]) {
                              q.push(v);
                              inQueue[v] = true;
                              count[v]++;
                              
                              if (count[v] >= n) {
                                  return true;
                              }
                          }
                      }
                  }
              }
              
              return false;
          }
          
          // 获取路径
          vector<int> getPath(int n, vector<vector<pair<int, int>>>& graph, int start, int end) {
              vector<int> dist(n, INT_MAX);
              vector<int> prev(n, -1);
              vector<bool> inQueue(n, false);
              vector<int> count(n, 0);
              
              dist[start] = 0;
              queue<int> q;
              q.push(start);
              inQueue[start] = true;
              count[start]++;
              
              while (!q.empty()) {
                  int u = q.front();
                  q.pop();
                  inQueue[u] = false;
                  
                  for (auto& [v, weight] : graph[u]) {
                      if (dist[u] != INT_MAX && dist[u] + weight < dist[v]) {
                          dist[v] = dist[u] + weight;
                          prev[v] = u;
                          
                          if (!inQueue[v]) {
                              q.push(v);
                              inQueue[v] = true;
                              count[v]++;
                              
                              if (count[v] > n) {
                                  return {};  // 存在负环
                              }
                          }
                      }
                  }
              }
              
              // 重构路径
              if (dist[end] == INT_MAX) return {};
              
              vector<int> path;
              for (int at = end; at != -1; at = prev[at]) {
                  path.push_back(at);
              }
              reverse(path.begin(), path.end());
              return path;
          }
      };
      class FloydWarshall {
      public:
          // 返回所有节点对的最短距离
          vector<vector<int>> allPairsShortestPath(int n, vector<vector<int>>& edges) {
              // 初始化距离矩阵
              vector<vector<int>> dist(n, vector<int>(n, INT_MAX));
              
              // 对角线为0
              for (int i = 0; i < n; i++) {
                  dist[i][i] = 0;
              }
              
              // 添加边
              for (auto& edge : edges) {
                  int u = edge[0], v = edge[1], w = edge[2];
                  dist[u][v] = min(dist[u][v], w);
              }
              
              // Floyd-Warshall核心算法
              for (int k = 0; k < n; k++) {
                  for (int i = 0; i < n; i++) {
                      if (dist[i][k] == INT_MAX) continue;
                      for (int j = 0; j < n; j++) {
                          if (dist[k][j] == INT_MAX) continue;
                          if (dist[i][k] + dist[k][j] < dist[i][j]) {
                              dist[i][j] = dist[i][k] + dist[k][j];
                          }
                      }
                  }
              }
              
              // 检查负环
              for (int i = 0; i < n; i++) {
                  if (dist[i][i] < 0) {
                      return {};  // 存在负环
                  }
              }
              
              return dist;
          }
          
          // 获取路径
          vector<vector<int>> getPaths(int n, vector<vector<int>>& edges) {
              vector<vector<int>> dist(n, vector<int>(n, INT_MAX));
              vector<vector<int>> next(n, vector<int>(n, -1));
              
              // 初始化
              for (int i = 0; i < n; i++) {
                  dist[i][i] = 0;
                  next[i][i] = i;
              }
              
              // 添加边
              for (auto& edge : edges) {
                  int u = edge[0], v = edge[1], w = edge[2];
                  if (w < dist[u][v]) {
                      dist[u][v] = w;
                      next[u][v] = v;
                  }
              }
              
              // Floyd-Warshall
              for (int k = 0; k < n; k++) {
                  for (int i = 0; i < n; i++) {
                      if (dist[i][k] == INT_MAX) continue;
                      for (int j = 0; j < n; j++) {
                          if (dist[k][j] == INT_MAX) continue;
                          if (dist[i][k] + dist[k][j] < dist[i][j]) {
                              dist[i][j] = dist[i][k] + dist[k][j];
                              next[i][j] = next[i][k];
                          }
                      }
                  }
              }
              
              return next;
          }
          
          // 重构i到j的路径
          vector<int> reconstructPath(int i, int j, const vector<vector<int>>& next) {
              if (next[i][j] == -1) return {};
              
              vector<int> path;
              path.push_back(i);
              
              while (i != j) {
                  i = next[i][j];
                  path.push_back(i);
              }
              
              return path;
          }
      };
      ```
    *   **

```C++
// ST表静态区间查询最值RMQ
class SparseTable {
private:
    vector<vector<int>> st; // st[i][j] 表示从i开始长度为2^j的区间的最值
    vector<int> log; // 预处理log2，方便查询
    int n;
    bool maxQuery; // true表示查询最大值，false表示查询最小值

    int combine(int a, int b) {
        return maxQuery ? max(a, b) : min(a, b);
    }

public:
    SparseTable(const vector<int>& arr, bool isMax = true) : maxQuery(isMax) {
        n = arr.size();
        int k = log2(n) + 1;
        st.assign(n, vector<int>(k));

        // 初始化长度为1的区间
        for (int i = 0; i < n; i++) {
            st[i][0] = arr[i];
        }

        // 动态规划预处理
        for (int j = 1; (1 << j) <= n; j++) {
            for (int i = 0; i + (1 << j) - 1 < n; i++) {
                st[i][j] = combine(st[i][j-1], st[i + (1 << (j-1))][j-1]);
            }
        }

        // 预处理log2
        log.resize(n + 1);
        log[1] = 0;
        for (int i = 2; i <= n; i++) {
            log[i] = log[i/2] + 1;
        }
    }

    // 查询区间[l, r]的最值，区间从0开始
    int query(int l, int r) {
        int j = log[r - l + 1];
        return combine(st[l][j], st[r - (1 << j) + 1][j]);
    }
};
```

```C++
// LCA
class LCA {
private:
    int n;
    int LOG;
    vector<vector<int>> adj; // 邻接表
    vector<vector<int>> up;  // up[i][j]表示节点i的第2^j级祖先
    vector<int> depth;       // 节点深度

public:
    LCA(int nodes, const vector<vector<int>>& graph) : n(nodes), adj(graph) {
        LOG = log2(n) + 1;
        up.assign(n, vector<int>(LOG, -1));
        depth.resize(n);
        // 假设根节点为0，进行初始化
        dfs(0, -1);
    }

    void dfs(int u, int parent) {
        up[u][0] = parent;
        for (int i = 1; i < LOG; i++) {
            if (up[u][i-1] != -1) {
                up[u][i] = up[up[u][i-1]][i-1];
            } else {
                up[u][i] = -1;
            }
        }

        for (int v : adj[u]) {
            if (v != parent) {
                depth[v] = depth[u] + 1;
                dfs(v, u);
            }
        }
    }

    // 将节点u向上移动k步
    int lift(int u, int k) {
        for (int i = 0; i < LOG; i++) {
            if (k & (1 << i)) {
                u = up[u][i];
                if (u == -1) break;
            }
        }
        return u;
    }

    int lca(int u, int v) {
        if (depth[u] < depth[v]) swap(u, v);
        // 将u提升到与v同一深度
        u = lift(u, depth[u] - depth[v]);

        if (u == v) return u;

        // 二进制提升，同时向上跳
        for (int i = LOG - 1; i >= 0; i--) {
            if (up[u][i] != up[v][i])
```

```C++
// DP相关 (一般先用一维数组)
DP问题：
构建一般的 DP 数组的关键在于理解：
行代表决策阶段（考虑到第几个物品），
列代表资源限制（背包容量/时间限制），
格子里的值代表目标（最大价值）。
int main()
{
	int r;
	cin >> r;
	vector<vector<int>> dp(r,vector<int>(r));
	for (int i = 1;i <= r;i++)
	{
		for (int j = 1;j <= i;j++)
		{
			cin >> dp[i-1][j-1];
		}
	}
	
	for (int i = 1;i < r;i++)
	{
		for (int j = 0;j <= i;j++)
		{
			if (j == 0) dp[i][j] += dp[i-1][j];
			else if (j == i) dp[i][j] += dp[i-1][j-1];
			else dp[i][j] += max(dp[i-1][j-1] , dp[i-1][j]);
		}
	}
	vector<int> last_row = dp[r-1];
    sort(last_row.begin(), last_row.end());
    
    cout << last_row[r-1] << endl;
	
	return 0;
}

0/1背包问题
数组的值 dp[i][j] 代表：
“在只有 j 这么多时间的情况下，
从前 i 株草药里随便挑，能得到的最大总价值。”
#include <bits/stdc++.h>
using namespace std;

int main()
{
	int t,m;
	cin >>t >>m;
	vector <vector<int>> dp(m+1,vector<int> (t+1,0));
	
	vector <int> cost(m+1);
	vector <int> value(m+1);
	
	for (int i = 1;i<=m;i++)
	{
		cin >> cost[i] >> value[i];
	}
	
	for (int i = 1;i <= m;i++)
	{
		for (int j = 0;j <= t;j++)
		{
			if(j >= cost[i]) dp[i][j] = max(dp[i-1][j],dp[i-1][j-cost[i]] + value[i]); 
			else dp[i][j] = dp[i-1][j];
		}
	}
	
	cout <<dp[m][t] << endl;
	return 0;
}

// 考虑用一维数组遍历，倒序遍历，正序遍历会变成完全背包问题
#include <bits/stdc++.h>
using namespace std;

int main()
{
	ios::sync_with_stdio(false);
	cin.tie(0);cout.tie(0);
	
	int n,m;
	cin >> n >> m;
	
	vector<int> dp(m+1,0);
	for (int i = 1; i<=n; i++)
	{
		int w,d;
		cin >> w>> d;
		for (int j = m; j >= w;j--)
		{
			dp[j] = max(dp[j],dp[j-w] + d);
		}
	}
	cout << dp[m] << endl;
	return 0;
}

0/1 背包： 每种物品只能拿 1 次（要么拿，要么不拿）。
多重背包： 每种物品有限制的数量（例如只能拿 5 次）。
完全背包： 每种物品可以拿无数次（只要你的空间/时间足够）。

在 0/1 背包中，内层循环必须从大到小（倒序），
为了防止同一个物品被加多次。
但在 完全背包中，因为我们要允许同一个物品被加多次,
所以内层循环必须从小到大（正序）。

#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

int main()
{
	ios::sync_with_stdio(false);
	cin.tie(0);cout.tie(0);
	
	int t,m;
	cin >> t >> m;
	vector<ll> dp(t+1);
	
	for (int i = 1; i <= m; i++)
	{
		int time;
		int value;
		cin >> time >> value;
		for (int j = time;j <= t;j++)
			dp[j] = max(dp[j],dp[j-time] + value);
	}
	
	cout << dp[t] << endl;
	return 0;
}

https://vjudge.net/contest/770035#problem/F
经典的多重背包问题
二进制拆分(针对物品的数量)
也就是把k件物品捆成一件物品放回去
转化为0/1背包问题
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

struct treasure
{
	int w;
	int v;
};
vector<treasure> goods;

int main()
{
	ios::sync_with_stdio(false);
	cin.tie(0);cout.tie(0);
	
	int n,W;
	cin >> n >> W;
	
	for (int i = 0;i < n;i++)
	{
		int v,w,m;
		cin >> v >> w >> m;
		for (int j = 1;j <= m;j*=2)
		{
			goods.push_back({w*j,v*j});
			m-=j;
		}
		if (m > 0) goods.push_back({w*m,v*m});
	}
	vector<ll> dp(W+1);
	for (size_t i = 0; i < goods.size();i++)
	{
		for (int j = W;j >= goods[i].w;j--)
			dp[j] = max(dp[j],dp[j - goods[i].w] + goods[i].v);
	}
	
	cout << dp[W] << endl;
	return 0;
}
```

```c++
// 线段树板子 (可以区间加，区间和)

typedef long long ll;
class segmenttree
{
private:
	vector<ll> tree;
	vector<ll> lazy;
	int n;
	
	void build(const vector<ll>& arr,int node,int start,int end)
	{
		if (start == end) tree[node] = arr[start];
		else
		{
			int mid = (start+end)/2;
			build(arr,2*node,start,mid);
			build(arr,2*node + 1,mid + 1,end);
			tree[node] = tree[2*node] + tree[2*node + 1];
		}
	}
	void push(int node,int start, int end)
	{
		if (lazy[node] != 0) 
		{
			tree[node] += lazy[node] * (end-start+1);
			if (start != end)
			{
				lazy[2*node] += lazy[node];
				lazy[2*node+1] += lazy[node];
			}
		}
		lazy[node] = 0;
	}
	void update(int node,int start,int end,int l,int r,ll val)
	{
		push(node,start,end);
		if (r < start || end < l) return;
		if (l <= start && end <= r)
		{
			lazy[node] += val;
			push(node,start,end);
			return;
		}
		int mid = (start + end)/2;
		update(2*node,start,mid,l,r,val);
		update(2*node+1,mid+1,end,l,r,val);
		tree[node] = tree[2*node] + tree[2*node+1];
	}
	
	ll query(int node,int start,int end,int l,int r)
	{
		push(node,start,end);
		if (r < start || end < l) return 0;
		if (l <= start && end <= r) return tree[node];
		int mid = (start + end)/2;
		ll left_sum = query(2*node,start,mid,l,r);
		ll right_sum = query(2*node+1,mid+1,end,l,r);
		return left_sum + right_sum;
	}
	
public:
	segmenttree(const vector<ll>& arr)
	{
		n = arr.size();
		tree.resize(4*n);
		lazy.resize(4*n,0);
		build(arr,1,0,n-1);
	}
	
	void update(int l,int r,ll val)
	{
		update(1,0,n-1,l,r,val);
	}
	
	ll query(int l,int r)
	{
		return query(1,0,n-1,l,r);
	}
};

int main()
{
	ios::sync_with_stdio(false);
	cin.tie(0);
	
	int n,m;
	cin >>n >>m;
	
	vector<ll> arr(n);
	for (int i = 0;i<n;i++)
	{
		cin>>arr[i];
	}
	
	segmenttree st(arr);
	
	while(m--)
	{
		int op;
		cin >> op;
		if (op == 1)
		{
			int x,y;
			ll k;
			cin >> x>> y >> k;
			st.update(x-1,y-1,k);
		}
		else if (op == 2)
		{
			int x,y;
			cin >> x >> y;
			ll res = st.query(x-1,y-1);
			cout << res << endl;
		}
	}
	return 0;
}

// 下面比上一个板子多实现了一个乘法操作
const int MAXN = 100005;
long long n, q, m;
long long a[MAXN];
long long sum[MAXN * 4], mul[MAXN * 4], add[MAXN * 4];

#define ls (p << 1)
#define rs (p << 1 | 1)

void push_up(int p) 
{
    sum[p] = (sum[ls] + sum[rs]) % m;
}

void build(int p, int l, int r) 
{
    mul[p] = 1;
    add[p] = 0;
    if (l == r) 
    {
        sum[p] = a[l] % m;
        return;
    }
    int mid = (l + r) >> 1;
    build(ls, l, mid);
    build(rs, mid + 1, r);
    push_up(p);
}

void push_down(int p, int l, int r) 
{
    if (mul[p] == 1 && add[p] == 0) return;

    int mid = (l + r) >> 1;
    
    sum[ls] = (sum[ls] * mul[p] + add[p] * (mid - l + 1)) % m;
    mul[ls] = (mul[ls] * mul[p]) % m;
    add[ls] = (add[ls] * mul[p] + add[p]) % m;
    
    sum[rs] = (sum[rs] * mul[p] + add[p] * (r - mid)) % m;
    mul[rs] = (mul[rs] * mul[p]) % m;
    add[rs] = (add[rs] * mul[p] + add[p]) % m;
    
    mul[p] = 1;
    add[p] = 0;
}

void update_mul(int p, int l, int r, int x, int y, long long k) 
{
    if (x <= l && r <= y) 
    {
        sum[p] = (sum[p] * k) % m;
        mul[p] = (mul[p] * k) % m;
        add[p] = (add[p] * k) % m;
        return;
    }
    push_down(p, l, r);
    int mid = (l + r) >> 1;
    if (x <= mid) update_mul(ls, l, mid, x, y, k);
    if (y > mid)  update_mul(rs, mid + 1, r, x, y, k);
    push_up(p);
}

void update_add(int p, int l, int r, int x, int y, long long k) 
{
    if (x <= l && r <= y) 
    {
        sum[p] = (sum[p] + k * (r - l + 1)) % m;
        add[p] = (add[p] + k) % m;
        return;
    }
    push_down(p, l, r);
    int mid = (l + r) >> 1;
    if (x <= mid) update_add(ls, l, mid, x, y, k);
    if (y > mid)  update_add(rs, mid + 1, r, x, y, k);
    push_up(p);
}

long long query(int p, int l, int r, int x, int y) 
{
    if (x <= l && r <= y) return sum[p];
    push_down(p, l, r);
    int mid = (l + r) >> 1;
    long long res = 0;
    if (x <= mid) res = (res + query(ls, l, mid, x, y)) % m;
    if (y > mid)  res = (res + query(rs, mid + 1, r, x, y)) % m;
    return res;
}

int main() 
{
    ios::sync_with_stdio(false);
    cin.tie(0);

    cin >> n >> q >> m;
    for (int i = 1; i <= n; i++) cin >> a[i];

    build(1, 1, n);

    while (q--) 
    {
        int op;
        cin >> op;
        if (op == 1) 
        {
            int x, y;
            long long k;
            cin >> x >> y >> k;
            update_mul(1, 1, n, x, y, k);
        } 
        else if (op == 2) 
        { 
            int x, y;
            long long k;
            cin >> x >> y >> k;
            update_add(1, 1, n, x, y, k);
        } 
        else 
        {
            int x, y;
            cin >> x >> y;
            cout << query(1, 1, n, x, y) << "\n";
        }
    }
    return 0;
}

树状数组的板子BIT
单点修改 + 区间求和
const int N = 5e5 + 5;
vector <int> tree(N);
int n,m;

int lowbit(int x)
{
	return x & -x;
}

void add(int x,int k)
{
	for (;x <= n;x += lowbit(x))
		tree[x] += k;
}

int ask(int x)
{
	int sum = 0;
	for (;x > 0;x -= lowbit(x))
	 	sum += tree[x];
	return sum;
}

int main()
{
	cin >>n >>m;
	for (int i = 1;i <= n;i++)
	{
		int val;
		cin >>val;
		add(i,val);
	}
	
	for (int i = 0;i < m;i++)
	{
		int op;
		cin >> op;
		if (op == 1)
		{
			int x,k;
			cin >>x >>k;
			add(x,k);
		}
		if (op == 2)
		{
			int l,r;
			cin >>l >>r;
			cout << ask(r) - ask(l-1) << endl;
		}
	}
	
	return 0;
}

区间修改 + 单点查询
// 前面都不变
int main()
{
	ios::sync_with_stdio(false);
	cin.tie(0); cout.tie(0);
	
	cin >>n >>m;
	
	int last = 0;
	for (int i = 1;i <= n;i++)
	{
		int current;
		cin >> current;
		add(i,current-last);
		last = current;
	}
	
	for (int i = 0;i < m;i++)
	{
		int op;
		cin >> op;
		if (op == 1)
		{
			int r,l,k;
			cin >> r >> l >> k;
			add(r,k);
			add(l+1,-k);
		}
		if (op == 2)
		{
			int x;
			cin >> x;
			cout << ask(x) << endl;
		}
	}
	
	return 0;
```

```c++
// 线性DP
class LIS {
public:
    int lengthOfLIS(vector<int>& nums) {
        if (nums.empty()) return 0;
        int n = nums.size();
        vector<int> dp(n, 1); 
        
        int maxLen = 1;
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = max(dp[i], dp[j] + 1);
                }
            }
            maxLen = max(maxLen, dp[i]);
        }
        return maxLen;
    }
    
    int lengthOfLIS_Optimized(vector<int>& nums) {
        vector<int> tails;  
        
        for (int num : nums) {
            auto it = lower_bound(tails.begin(), tails.end(), num);
            if (it == tails.end()) {
                tails.push_back(num);  
            } else {
                *it = num;  
            }
        }
        return tails.size();
    }
    
    vector<int> getLIS(vector<int>& nums) {
        int n = nums.size();
        vector<int> tails;
        vector<int> parent(n, -1);
        vector<int> indices;  
        
        for (int i = 0; i < n; i++) {
            auto it = lower_bound(tails.begin(), tails.end(), nums[i]);
            int pos = it - tails.begin();
            
            if (it == tails.end()) {
                tails.push_back(nums[i]);
                indices.push_back(i);
            } else {
                *it = nums[i];
                indices[pos] = i;
            }
            
            if (pos > 0) {
                parent[i] = indices[pos - 1];
            }
        }
        
        vector<int> lis;
        int idx = indices.back();
        while (idx != -1) {
            lis.push_back(nums[idx]);
            idx = parent[idx];
        }
        reverse(lis.begin(), lis.end());
        return lis;
    }
};
class LCS {
public:
    int longestCommonSubsequence(string text1, string text2) {
        int m = text1.size(), n = text2.size();
        vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
        
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (text1[i - 1] == text2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[m][n];
    }
    
    int longestCommonSubsequence_Optimized(string text1, string text2) {
        if (text1.size() < text2.size()) swap(text1, text2);
        int m = text1.size(), n = text2.size();
        vector<int> dp(n + 1, 0);
        
        for (int i = 1; i <= m; i++) {
            int prev = 0;  
            for (int j = 1; j <= n; j++) {
                int temp = dp[j];  
                if (text1[i - 1] == text2[j - 1]) {
                    dp[j] = prev + 1;
                } else {
                    dp[j] = max(dp[j], dp[j - 1]);
                }
                prev = temp;  
            }
        }
        return dp[n];
    }

    string getLCS(string text1, string text2) {
        int m = text1.size(), n = text2.size();
        vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
        
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (text1[i - 1] == text2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        
        string lcs;
        int i = m, j = n;
        while (i > 0 && j > 0) {
            if (text1[i - 1] == text2[j - 1]) {
                lcs = text1[i - 1] + lcs;
                i--; j--;
            } else if (dp[i - 1][j] > dp[i][j - 1]) {
                i--;
            } else {
                j--;
            }
        }
        return lcs;
    }
};
class EditDistance {
public:
    int minDistance(string word1, string word2) {
        int m = word1.size(), n = word2.size();
        vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
        
        for (int i = 0; i <= m; i++) dp[i][0] = i;  
        for (int j = 0; j <= n; j++) dp[0][j] = j;  
        
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1[i - 1] == word2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1];  
                } else {
                    dp[i][j] = min({
                        dp[i][j - 1] + 1,      
                        dp[i - 1][j] + 1,      
                        dp[i - 1][j - 1] + 1   
                    });
                }
            }
        }
        return dp[m][n];
    }
    
    int minDistance_Optimized(string word1, string word2) {
        if (word1.size() < word2.size()) swap(word1, word2);
        int m = word1.size(), n = word2.size();
        vector<int> dp(n + 1, 0);
        
        for (int j = 0; j <= n; j++) dp[j] = j;
        
        for (int i = 1; i <= m; i++) {
            int prev = dp[0];  
            dp[0] = i;
            
            for (int j = 1; j <= n; j++) {
                int temp = dp[j];
                if (word1[i - 1] == word2[j - 1]) {
                    dp[j] = prev;
                } else {
                    dp[j] = min({dp[j - 1], dp[j], prev}) + 1;
                }
                prev = temp;
            }
        }
        return dp[n];
    }
};
// 区间DP
class StoneMerge {
public:
    // 环形石子合并 - 最小合并代价
    int minCost(vector<int>& stones) {
        int n = stones.size();
        if (n == 0) return 0;
        
        // 环形转链：复制一倍
        vector<int> prefix(2 * n + 1, 0);
        for (int i = 1; i <= 2 * n; i++) {
            prefix[i] = prefix[i - 1] + stones[(i - 1) % n];
        }
        
        vector<vector<int>> dp(2 * n, vector<int>(2 * n, 0));
        
        // 枚举区间长度
        for (int len = 2; len <= n; len++) {
            for (int l = 0; l + len - 1 < 2 * n; l++) {
                int r = l + len - 1;
                dp[l][r] = INT_MAX;
                
                // 枚举分割点
                for (int k = l; k < r; k++) {
                    int cost = dp[l][k] + dp[k + 1][r] + prefix[r + 1] - prefix[l];
                    dp[l][r] = min(dp[l][r], cost);
                }
            }
        }
        
        // 找出最小代价（长度为n的区间）
        int minCost = INT_MAX;
        for (int i = 0; i < n; i++) {
            minCost = min(minCost, dp[i][i + n - 1]);
        }
        return minCost;
    }
    
    // 直线型石子合并
    int minCostLinear(vector<int>& stones) {
        int n = stones.size();
        vector<int> prefix(n + 1, 0);
        for (int i = 1; i <= n; i++) {
            prefix[i] = prefix[i - 1] + stones[i - 1];
        }
        
        vector<vector<int>> dp(n, vector<int>(n, 0));
        
        for (int len = 2; len <= n; len++) {
            for (int l = 0; l + len - 1 < n; l++) {
                int r = l + len - 1;
                dp[l][r] = INT_MAX;
                
                for (int k = l; k < r; k++) {
                    dp[l][r] = min(dp[l][r], 
                                   dp[l][k] + dp[k + 1][r] + prefix[r + 1] - prefix[l]);
                }
            }
        }
        return dp[0][n - 1];
    }
    
    // 最大合并代价（适用于某些变体问题）
    int maxCost(vector<int>& stones) {
        int n = stones.size();
        vector<int> prefix(2 * n + 1, 0);
        for (int i = 1; i <= 2 * n; i++) {
            prefix[i] = prefix[i - 1] + stones[(i - 1) % n];
        }
        
        vector<vector<int>> dp(2 * n, vector<int>(2 * n, 0));
        
        for (int len = 2; len <= n; len++) {
            for (int l = 0; l + len - 1 < 2 * n; l++) {
                int r = l + len - 1;
                
                for (int k = l; k < r; k++) {
                    dp[l][r] = max(dp[l][r], 
                                  dp[l][k] + dp[k + 1][r] + prefix[r + 1] - prefix[l]);
                }
            }
        }
        
        int maxCost = 0;
        for (int i = 0; i < n; i++) {
            maxCost = max(maxCost, dp[i][i + n - 1]);
        }
        return maxCost;
    }
};
class MatrixChainMultiplication {
public:
    // 返回最小乘法次数
    int minMultiplications(vector<pair<int, int>>& matrices) {
        int n = matrices.size();
        vector<vector<int>> dp(n, vector<int>(n, 0));
        
        // len表示链的长度
        for (int len = 2; len <= n; len++) {
            for (int i = 0; i + len - 1 < n; i++) {
                int j = i + len - 1;
                dp[i][j] = INT_MAX;
                
                // 枚举分割点
                for (int k = i; k < j; k++) {
                    int cost = dp[i][k] + dp[k + 1][j] +
                              matrices[i].first * matrices[k].second * matrices[j].second;
                    dp[i][j] = min(dp[i][j], cost);
                }
            }
        }
        return dp[0][n - 1];
    }
    
    // 返回最优括号化方案
    string getOptimalParenthesization(vector<pair<int, int>>& matrices) {
        int n = matrices.size();
        vector<vector<int>> dp(n, vector<int>(n, 0));
        vector<vector<int>> split(n, vector<int>(n, -1));
        
        for (int len = 2; len <= n; len++) {
            for (int i = 0; i + len - 1 < n; i++) {
                int j = i + len - 1;
                dp[i][j] = INT_MAX;
                
                for (int k = i; k < j; k++) {
                    int cost = dp[i][k] + dp[k + 1][j] +
                              matrices[i].first * matrices[k].second * matrices[j].second;
                    if (cost < dp[i][j]) {
                        dp[i][j] = cost;
                        split[i][j] = k;
                    }
                }
            }
        }
        
        // 递归构造括号化方案
        function<string(int, int)> build = [&](int i, int j) -> string {
            if (i == j) {
                return "A" + to_string(i + 1);
            }
            int k = split[i][j];
            string left = build(i, k);
            string right = build(k + 1, j);
            return "(" + left + " × " + right + ")";
        };
        
        return build(0, n - 1);
    }
};
class PalindromeDP {
public:
    // 最长回文子序列
    int longestPalindromeSubseq(string s) {
        int n = s.size();
        vector<vector<int>> dp(n, vector<int>(n, 0));
        
        // 初始化：单个字符是回文
        for (int i = 0; i < n; i++) {
            dp[i][i] = 1;
        }
        
        // 区间DP
        for (int len = 2; len <= n; len++) {
            for (int i = 0; i + len - 1 < n; i++) {
                int j = i + len - 1;
                if (s[i] == s[j]) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                } else {
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[0][n - 1];
    }
    
    // 最少插入次数构成回文串
    int minInsertionsToPalindrome(string s) {
        int n = s.size();
        vector<vector<int>> dp(n, vector<int>(n, 0));
        
        for (int len = 2; len <= n; len++) {
            for (int i = 0; i + len - 1 < n; i++) {
                int j = i + len - 1;
                if (s[i] == s[j]) {
                    dp[i][j] = dp[i + 1][j - 1];
                } else {
                    dp[i][j] = min(dp[i + 1][j], dp[i][j - 1]) + 1;
                }
            }
        }
        return dp[0][n - 1];
    }
    
    // 判断回文子串 - Manacher算法 O(n)
    string longestPalindrome(string s) {
        if (s.empty()) return "";
        
        // 预处理，插入分隔符
        string t = "#";
        for (char c : s) {
            t += c;
            t += '#';
        }
        
        int n = t.size();
        vector<int> p(n, 0);  // p[i]表示以i为中心的最长回文半径
        int center = 0, right = 0;
        int maxLen = 0, maxCenter = 0;
        
        for (int i = 0; i < n; i++) {
            // 利用对称性
            int mirror = 2 * center - i;
            if (i < right) {
                p[i] = min(right - i, p[mirror]);
            }
            
            // 尝试扩展
            int a = i + (1 + p[i]);
            int b = i - (1 + p[i]);
            while (a < n && b >= 0 && t[a] == t[b]) {
                p[i]++;
                a++;
                b--;
            }
            
            // 更新中心和右边界
            if (i + p[i] > right) {
                center = i;
                right = i + p[i];
            }
            
            if (p[i] > maxLen) {
                maxLen = p[i];
                maxCenter = i;
            }
        }
        
        // 还原原字符串
        int start = (maxCenter - maxLen) / 2;
        return s.substr(start, maxLen);
    }
};
```

```C++
// 位运算
class BitOperations {
public:
    // 判断奇偶
    bool isOdd(int n) {
        return n & 1;  // 奇数为1，偶数为0
    }
    
    // 判断是否是2的幂
    bool isPowerOfTwo(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }
    
    // 获取最低位的1
    int lowbit(int n) {
        return n & -n;  // 返回最低位1及其后面的0
    }
    
    // 移除最低位的1
    int removeLowbit(int n) {
        return n & (n - 1);
    }
    
    // 统计1的个数（汉明重量）
    int popcount(int n) {
        int count = 0;
        while (n) {
            n &= n - 1;  // 清除最低位的1
            count++;
        }
        return count;
    }
    
    // 快速幂 (a^b mod m)
    long long fastPow(long long a, long long b, long long m = 1e9 + 7) {
        long long res = 1;
        a %= m;
        while (b > 0) {
            if (b & 1) res = (res * a) % m;
            a = (a * a) % m;
            b >>= 1;
        }
        return res;
    }
    
    // 循环左移
    unsigned int rotateLeft(unsigned int n, int k) {
        return (n << k) | (n >> (32 - k));
    }
    
    // 循环右移
    unsigned int rotateRight(unsigned int n, int k) {
        return (n >> k) | (n << (32 - k));
    }
    
    // 反转二进制位
    uint32_t reverseBits(uint32_t n) {
        n = ((n & 0xAAAAAAAA) >> 1) | ((n & 0x55555555) << 1);  // 交换相邻位
        n = ((n & 0xCCCCCCCC) >> 2) | ((n & 0x33333333) << 2);  // 交换每2位
        n = ((n & 0xF0F0F0F0) >> 4) | ((n & 0x0F0F0F0F) << 4);  // 交换每4位
        n = ((n & 0xFF00FF00) >> 8) | ((n & 0x00FF00FF) << 8);  // 交换每8位
        n = ((n & 0xFFFF0000) >> 16) | ((n & 0x0000FFFF) << 16); // 交换每16位
        return n;
    }
};
class BitmaskCombinations {
public:
    // 枚举所有子集
    vector<vector<int>> getAllSubsets(vector<int>& nums) {
        int n = nums.size();
        vector<vector<int>> result;
        
        for (int mask = 0; mask < (1 << n); mask++) {
            vector<int> subset;
            for (int i = 0; i < n; i++) {
                if (mask & (1 << i)) {
                    subset.push_back(nums[i]);
                }
            }
            result.push_back(subset);
        }
        return result;
    }
    
    // 枚举大小为k的所有子集（Gosper's Hack）
    vector<vector<int>> getSubsetsOfSizeK(vector<int>& nums, int k) {
        int n = nums.size();
        vector<vector<int>> result;
        
        if (k == 0) {
            result.push_back({});
            return result;
        }
        
        int mask = (1 << k) - 1;  // 最小的k个1
        while (mask < (1 << n)) {
            // 获取当前掩码对应的子集
            vector<int> subset;
            for (int i = 0; i < n; i++) {
                if (mask & (1 << i)) {
                    subset.push_back(nums[i]);
                }
            }
            result.push_back(subset);
            
            // Gosper's Hack: 生成下一个包含k个1的掩码
            int c = mask & -mask;
            int r = mask + c;
            mask = (((r ^ mask) >> 2) / c) | r;
        }
        return result;
    }
    
    // 判断两个集合是否有交集
    bool hasIntersection(int mask1, int mask2) {
        return (mask1 & mask2) != 0;
    }
    
    // 计算集合的并集
    int unionMask(int mask1, int mask2) {
        return mask1 | mask2;
    }
    
    // 计算集合的交集
    int intersectionMask(int mask1, int mask2) {
        return mask1 & mask2;
    }
    
    // 计算集合的差集 (mask1 - mask2)
    int differenceMask(int mask1, int mask2) {
        return mask1 & (~mask2);
    }
    
    // 获取所有超集
    vector<int> getAllSupersets(int mask, int n) {
        vector<int> supersets;
        int base = mask;
        int remaining = ((1 << n) - 1) ^ mask;  // mask的补集
        
        // 枚举remaining的所有子集
        int sub = 0;
        do {
            supersets.push_back(base | sub);
            sub = (sub - remaining) & remaining;
        } while (sub != 0);
        
        return supersets;
    }
    
    // 动态规划：子集和问题（恰好装满）
    bool subsetSum(vector<int>& nums, int target) {
        int n = nums.size();
        int m = 1 << n;
        vector<bool> dp(m, false);
        dp[0] = true;
        
        for (int mask = 0; mask < m; mask++) {
            if (!dp[mask]) continue;
            
            for (int i = 0; i < n; i++) {
                if (mask & (1 << i)) continue;  // 已经选过
                
                int newMask = mask | (1 << i);
                dp[newMask] = true;
            }
        }
        
        // 检查所有子集的和
        for (int mask = 0; mask < m; mask++) {
            if (!dp[mask]) continue;
            
            int sum = 0;
            for (int i = 0; i < n; i++) {
                if (mask & (1 << i)) {
                    sum += nums[i];
                }
            }
            if (sum == target) return true;
        }
        return false;
    }
};
```

```c++
// 最小生成树
#include <bits/stdc++.h>
using namespace std;

struct edge
{
	int u,v,w;
	bool operator<(const edge& other) const
	{
		return w < other.w;
	}  
};

vector<int> parent;

int find(int x)
{
	if (parent[x] != x)
	{
		parent[x] = find(parent[x]);  
	}
	return parent[x];
}

bool unite(int x,int y)  
{
	int rx = find(x), ry = find(y);
	if (rx == ry) return false;
	parent[rx] = ry;
	return true;
}

int main()
{
	ios::sync_with_stdio(false);
	cin.tie(0);
	
	int N,M;
	cin >>N >>M;
	
	vector<edge> edges(M);
	for (int i = 0;i < M;i++)
	{
		cin >> edges[i].u >> edges[i].v >> edges[i].w;
	}
	
	sort(edges.begin(),edges.end()); 
	
	parent.resize(N+1);
	for(int j = 1;j <= N;j++)
	{
		parent[j] = j;  
	}
	
	long long weight = 0;
	int cnt = 0;
	
	for (const auto& e:edges)
	{
		if (unite(e.u,e.v))
		{
			weight += e.w;
			cnt ++;
			if (cnt == N-1) break;
		}
	}
	
	if (cnt == N-1)   cout << weight << endl;
	else   cout << "orz" << endl;
	
	return 0;
}
```

```c++
class BinaryAnswer {
public:
    // 浮点数二分答案模板
    double binarySearchDouble(double left, double right, double eps, function<bool(double)> check) {
        while (right - left > eps) {
            double mid = left + (right - left) / 2;
            if (check(mid)) {
                right = mid;  // 向左搜索
            } else {
                left = mid;   // 向右搜索
            }
        }
        return left;
    }
    
    // 整数二分答案模板（最小化最大值问题）
    int binarySearchMinMax(vector<int>& nums, int k, function<bool(int)> check) {
        int left = *max_element(nums.begin(), nums.end());  // 最小可能值
        int right = accumulate(nums.begin(), nums.end(), 0); // 最大可能值
        
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (check(mid)) {
                right = mid;  // 尝试更小的值
            } else {
                left = mid + 1;  // 需要更大的值
            }
        }
        return left;
    }
    
    // 整数二分答案模板（最大化最小值问题）
    int binarySearchMaxMin(vector<int>& nums, int k, function<bool(int)> check) {
        int left = *min_element(nums.begin(), nums.end());  // 最小可能值
        int right = *max_element(nums.begin(), nums.end()); // 最大可能值
        
        while (left < right) {
            int mid = left + (right - left + 1) / 2;  // 向上取整
            if (check(mid)) {
                left = mid;  // 尝试更大的值
            } else {
                right = mid - 1;  // 需要更小的值
            }
        }
        return left;
    }
    
    // 应用示例：在有序矩阵中搜索
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        if (matrix.empty() || matrix[0].empty()) return false;
        
        int m = matrix.size(), n = matrix[0].size();
        int left = 0, right = m * n - 1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            int row = mid / n;
            int col = mid % n;
            
            if (matrix[row][col] == target) {
                return true;
            } else if (matrix[row][col] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return false;
    }
    
    // 应用示例：寻找峰值元素
    int findPeakElement(vector<int>& nums) {
        int left = 0, right = nums.size() - 1;
        
        while (left < right) {
            int mid = left + (right - left) / 2;
            
            if (nums[mid] > nums[mid + 1]) {
                // 峰值在左侧
                right = mid;
            } else {
                // 峰值在右侧
                left = mid + 1;
            }
        }
        
        return left;
    }
};
```

