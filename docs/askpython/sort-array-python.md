# 如何在 Python 中对数组排序

> 原文：<https://www.askpython.com/python/array/sort-array-python>

Python 数组可以使用不同的排序算法进行排序，根据所选的算法，它们的运行时间和效率会有所不同。我们研究了对数组元素进行排序的一些方法。

* * *

## 对 Python 可迭代对象使用 sorted()

Python 使用一些非常有效的算法来执行排序。例如，`sorted()`方法使用一种叫做 **Timsort** (它是插入排序和合并排序的组合)的算法来执行高度优化的排序。

任何 Python 可迭代对象，比如列表或数组，都可以使用这个方法进行排序。

```py
import array

# Declare a list type object
list_object = [3, 4, 1, 5, 2]

# Declare an integer array object
array_object = array.array('i', [3, 4, 1, 5, 2])

print('Sorted list ->', sorted(list_object))
print('Sorted array ->', sorted(array_object))

```

**输出:**

```py
Sorted list -> [1, 2, 3, 4, 5]
Sorted array -> [1, 2, 3, 4, 5]

```

* * *

## 实现合并排序和快速排序

在这里，我们研究两种在实际应用中常用的排序技术，即**合并排序**和**快速排序**算法。

### 1.合并排序算法

该算法使用自下而上的分治方法，首先将原始数组划分为子数组，然后合并单独排序的子数组以产生最终排序的数组。

在下面的代码片段中，`mergesort_helper()`方法实际上将数组拆分成子数组，而 perform_merge()方法将两个先前排序的数组合并成一个新的排序数组。

```py
import array

def mergesort(a, arr_type):
    def perform_merge(a, arr_type, start, mid, end):
        # Merges two previously sorted arrays
        # a[start:mid] and a[mid:end]
        tmp = array.array(arr_type, [i for i in a])
        def compare(tmp, i, j):
            if tmp[i] <= tmp[j]:
                i += 1
                return tmp[i-1]
            else:
                j += 1
                return tmp[j-1]
        i = start
        j = mid + 1
        curr = start
        while i<=mid or j<=end:
            if i<=mid and j<=end:
                if tmp[i] <= tmp[j]:
                    a[curr] = tmp[i]
                    i += 1
                else:
                    a[curr] = tmp[j]
                    j += 1
            elif i==mid+1 and j<=end:
                a[curr] = tmp[j]
                j += 1
            elif j == end+1 and i<=mid:
                a[curr] = tmp[i]
                i += 1
            elif i > mid and j > end:
                break
            curr += 1

    def mergesort_helper(a, arr_type, start, end):
        # Divides the array into two parts
        # recursively and merges the subarrays
        # in a bottom up fashion, sorting them
        # via Divide and Conquer
        if start < end:
            mergesort_helper(a, arr_type, start, (end + start)//2)
            mergesort_helper(a, arr_type, (end + start)//2 + 1, end)
            perform_merge(a, arr_type, start, (start + end)//2, end)

    # Sorts the array using mergesort_helper
    mergesort_helper(a, arr_type, 0, len(a)-1)

```

**测试用例**:

```py
a = array.array('i', [3, 1, 2, 4, 5, 1, 3, 12, 7, 6])
print('Before MergeSort ->', a)
mergesort(a, 'i')
print('After MergeSort ->', a)

```

**输出:**

```py
Before MergeSort -> array('i', [3, 1, 2, 4, 5, 1, 3, 12, 7, 6])
After MergeSort -> array('i', [1, 1, 2, 3, 3, 4, 5, 6, 7, 12])

```

* * *

### 2.快速排序算法

该算法也使用分治策略，但是使用自顶向下的方法，首先围绕一个 **pivot** 元素划分数组(这里，我们总是选择数组的最后一个元素作为 pivot)。

从而确保在每一步之后，枢轴都位于最终排序数组中的指定位置。

在确保数组围绕轴心被划分后(小于轴心的元素在左边，大于轴心的元素在右边)，我们继续对数组的其余部分应用`partition`函数，直到所有元素都在它们各自的位置，这时数组被完全排序。

**注意**:该算法还有其他选择枢纽元素的方法。一些变体选择中间元素作为中枢，而另一些变体使用随机选择策略作为中枢。

```py
def quicksort(a, arr_type):
    def do_partition(a, arr_type, start, end):
        # Performs the partitioning of the subarray a[start:end]

        # We choose the last element as the pivot
        pivot_idx = end
        pivot = a[pivot_idx]

        # Keep an index for the first partition
        # subarray (elements lesser than the pivot element)
        idx = start - 1

        def increment_and_swap(j):
            nonlocal idx
            idx += 1
            a[idx], a[j] = a[j], a[idx]

        [increment_and_swap(j) for j in range(start, end) if a[j] < pivot]

        # Finally, we need to swap the pivot (a[end] with a[idx+1])
        # since we have reached the position of the pivot in the actual
        # sorted array
        a[idx+1], a[end] = a[end], a[idx+1]

        # Return the final updated position of the pivot
        # after partitioning
        return idx+1

    def quicksort_helper(a, arr_type, start, end):
        if start < end:
            # Do the partitioning first and then go via
            # a top down divide and conquer, as opposed
            # to the bottom up mergesort
            pivot_idx = do_partition(a, arr_type, start, end)
            quicksort_helper(a, arr_type, start, pivot_idx-1)
            quicksort_helper(a, arr_type, pivot_idx+1, end)

    quicksort_helper(a, arr_type, 0, len(a)-1)

```

在这里，`quicksort_helper`方法执行分治法的步骤，而`do_partition`方法围绕轴心分割数组并返回轴心的位置，围绕轴心我们继续递归分割轴心前后的子数组，直到整个数组排序完毕。

**测试用例**:

```py
b = array.array('i', [3, 1, 2, 4, 5, 1, 3, 12, 7, 6])
print('Before QuickSort ->', b)
quicksort(b, 'i')
print('After QuickSort ->', b)

```

**输出:**

```py
Before QuickSort -> array('i', [3, 1, 2, 4, 5, 1, 3, 12, 7, 6])
After QuickSort -> array('i', [1, 1, 2, 3, 3, 4, 5, 6, 7, 12])

```

* * *

## 结论

在本文中，我们研究了在 Python 数组上执行排序的 MergeSort 和 QuickSort 算法，了解了如何以自顶向下和自底向上的方式使用分治法。我们还简要地看了一下该语言提供的对 iterables 进行排序的本机`sorted()`方法。

## 参考资料:

*   [Python.org 排序()函数](https://docs.python.org/3.7/library/functions.html#sorted)
*   [综合维基百科品种](https://en.wikipedia.org/wiki/Merge_sort)
*   [快速排序维基百科](https://en.wikipedia.org/wiki/Quicksort)