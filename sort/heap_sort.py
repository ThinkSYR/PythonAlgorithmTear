
def heapify(arr, max_n, i):
    large = i
    l = 2 * i + 1
    r = 2 * i + 2
    # left
    if (l < max_n and arr[l] > arr[large]):
        large = l
    # right
    if (r < max_n and arr[r] > arr[large]):
        large = r
    # swap
    if large != i:
        arr[large], arr[i] = arr[i], arr[large]
        heapify(arr, max_n, large)

  
def heapSort(arr):
    # build
    for i in range(len(arr)//2, -1, -1):
        heapify(arr, len(arr), i)
    print(arr)
    # sort
    for i in range(len(arr)-1, -1, -1):
        arr[i], arr[0] = arr[0], arr[i]
        # 在i之前进行堆化
        heapify(arr, i, 0)

  
arr = [ 12, 11, 13, 5, 6, 7] 
heapSort(arr) 
n = len(arr) 
print ("排序后", arr) 

