# DSBA_KMITL_Computer_Programming_Numpy


## สรุปเนื้อหาเกี่ยวกับ Arrays ใน NumPy (ภาษาไทย) - ฉบับละเอียดพร้อมอธิบายและตัวอย่าง

เอกสารนี้อธิบายเกี่ยวกับ Arrays ใน NumPy ซึ่งเป็นไลบรารีพื้นฐานของ Python สำหรับการคำนวณทางคณิตศาสตร์โดยเน้นการจัดการอาเรย์หลายมิติ พร้อมคำอธิบายและตัวอย่างประกอบอย่างละเอียด

**1. Arrays คืออะไร?**

Array คือโครงสร้างข้อมูลที่เก็บค่าข้อมูลที่มีชนิดข้อมูลเดียวกันในรูปแบบตาราง (grid) คิดง่ายๆ เหมือนกับเป็น matrix  การเข้าถึงข้อมูลใน array ทำได้โดยใช้ index ซึ่งเป็น tuple ของ integers  เริ่มจาก 0 สำหรับ array 1 มิติก็ใช้ index ตัวเดียว, array 2 มิติก็ใช้ index 2 ตัว (แถว, คอลัมน์) เป็นต้น  NumPy มีประสิทธิภาพสูงในการจัดการ array ขนาดใหญ่ และมีฟังก์ชันมากมายสำหรับการคำนวณทางคณิตศาสตร์  การใช้ list ใน Python ธรรมดาทำงานได้ช้ากว่ามากเมื่อเทียบกับ NumPy array  โดยเฉพาะกับข้อมูลขนาดใหญ่

```python
import numpy as np

# สร้าง array 1 มิติ
arr1d = np.array([1, 2, 3, 4, 5])
print(arr1d)  # Output: [1 2 3 4 5]
print(arr1d[0]) # Output: 1  (element แรก)
print(arr1d[4]) # Output: 5 (element สุดท้าย)

# สร้าง array 2 มิติ
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2d)
# Output:
# [[1 2 3]
#  [4 5 6]]
print(arr2d[0, 1])  # Output: 2 (element แถวที่ 0, คอลัมน์ที่ 1)
print(arr2d[1, 0]) # Output: 4 (element แถวที่ 1, คอลัมน์ที่ 0)

# shape ของ array บอกขนาดของ array ในแต่ละมิติ
print(arr1d.shape)  # Output: (5,) หมายถึง array 1 มิติ มี 5 elements
print(arr2d.shape)  # Output: (2, 3) หมายถึง array 2 มิติ มี 2 แถว 3 คอลัมน์
```

**2. การสร้าง Array**

NumPy มีฟังก์ชันมากมายสำหรับสร้าง array หลากหลายรูปแบบ

```python
import numpy as np

# np.array(): สร้างจาก list หรือ tuple
arr = np.array([1, 2, 3])

# np.zeros(): สร้าง array ที่มีค่าเป็น 0 ทั้งหมด
zeros_arr = np.zeros((2, 3))  # shape (2, 3)

# np.ones(): สร้าง array ที่มีค่าเป็น 1 ทั้งหมด
ones_arr = np.ones((3, 2), dtype=int) # กำหนด dtype เป็น int

# np.arange(): สร้าง array ที่มีค่าเป็นลำดับเลข
range_arr = np.arange(5, 10) # 5, 6, 7, 8, 9

# np.linspace(): สร้าง array ที่มีค่าอยู่ในช่วงที่กำหนด โดยแบ่งเป็นจำนวนเท่าๆกัน
linspace_arr = np.linspace(0, 1, 5) # สร้าง 5 ค่า ตั้งแต่ 0 ถึง 1  (0, 0.25, 0.5, 0.75, 1)

# np.random.rand(): สร้าง array ที่มีค่าสุ่มระหว่าง 0 ถึง 1 (uniform distribution)
random_arr = np.random.rand(2, 3) 

# np.random.randint(): สร้าง array ที่มีค่าสุ่มเป็นจำนวนเต็ม
random_int_arr = np.random.randint(1, 10, size=(2, 4)) # เลขสุ่มระหว่าง 1 ถึง 9, shape (2, 4)

# np.eye(): สร้าง identity matrix
identity_matrix = np.eye(3) # 3x3 identity matrix

# np.full(): สร้าง array ที่มีค่าเท่ากันทั้งหมด
full_arr = np.full((2, 2), 7)  # array ที่มีค่า 7 ทั้งหมด, shape (2, 2)

#  สร้าง array จาก existing array
x = np.array([1, 2, 3])
y = np.asarray(x) #  ถ้า x เป็น numpy array อยู่แล้ว  y จะชี้ไปที่ memory เดียวกับ x  (ไม่ copy)
z = np.array(x) # สร้าง array ใหม่โดย copy ค่าจาก x

# empty_like, zeros_like, ones_like, full_like
a = np.array([[1,2,3], [4,5,6]])
b = np.empty_like(a) # shape เหมือน a แต่ค่าจะเป็นอะไรก็ได้
c = np.zeros_like(a)  # shape และ dtype เหมือน a แต่ค่าเป็น 0
```

**3. Data Type Objects (dtype)**

dtype กำหนดชนิดข้อมูลใน array เช่น int32, float64, bool, U20 (Unicode string ความยาว 20 ตัวอักษร)  การกำหนด dtype ช่วยประหยัด memory และเพิ่มประสิทธิภาพ

```python
import numpy as np

# กำหนด dtype ตอนสร้าง array
int_arr = np.array([1, 2, 3], dtype=np.int32)
float_arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
bool_arr = np.array([True, False, True], dtype=bool)
string_arr = np.array(['Alice', 'Bob', 'Charlie'], dtype='U10') #  string ยาวสุด 10 ตัวอักษร

#  สร้าง structured array - เก็บข้อมูลหลายชนิดใน array เดียว
data = np.array([('Alice', 25, 1.75), ('Bob', 30, 1.80)],
                 dtype=[('name', 'U10'), ('age', 'i4'), ('height', 'f8')])

print(data) #  แสดง array ทั้งหมด
print(data['name'])  #  แสดงเฉพาะ field 'name'
print(data[0]['age']) #  แสดงอายุของ element แรก

#  ตรวจสอบ dtype
print(int_arr.dtype) # Output: int32
print(data.dtype) # Output: [('name', '<U10'), ('age', '<i4'), ('height', '<f8')]

#  เปลี่ยน dtype ของ array  (astype)
float_arr = int_arr.astype(float) #  เปลี่ยน int_arr เป็น float
print(float_arr.dtype) # Output: float64

```

**4. Array Indexing**

การเข้าถึงข้อมูลใน array ทำได้โดยใช้ index ซึ่งเริ่มต้นที่ 0 และสิ้นสุดที่ n-1 โดย n คือจำนวน element ในมิติที่กำหนด สามารถใช้ index ติดลบเพื่อเข้าถึงข้อมูลจากท้าย array

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

print(arr[0, 1])  # Output: 2 (element แถวที่ 0, คอลัมน์ที่ 1)
print(arr[1, -1]) # Output: 6  (element สุดท้ายในแถวที่ 1)

arr1d = np.array([1, 2, 3, 4, 5])
print(arr1d[-2]) #  Output: 4 (element ที่ 2 นับจากท้าย array)

#  แก้ไขค่าใน array
arr[0, 0] = 10
print(arr) # array([[10,  2,  3], [ 4,  5,  6]])

```

**5. Boolean Array Indexing**

Boolean array indexing อนุญาตให้เลือก element ใน array ที่ตรงตามเงื่อนไขที่กำหนด  เป็นเทคนิคที่มีประโยชน์มากในการ filter ข้อมูล

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

#  สร้าง boolean array จากเงื่อนไข
bool_arr = arr > 2 
print(bool_arr) # [False False  True  True  True]

# ใช้ boolean array เพื่อเลือก element
filtered_arr = arr[bool_arr]
print(filtered_arr) # [3 4 5]

#  ทำในขั้นตอนเดียว
filtered_arr = arr[arr > 2]
print(filtered_arr) # [3 4 5]

#  เงื่อนไขซับซ้อน
filtered_arr = arr[(arr > 2) & (arr < 5)]  #  และ (&) หรือ (|)
print(filtered_arr) # [3 4]

#  ใช้กับ array หลายมิติ
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
filtered_arr2d = arr2d[arr2d > 3]  #  เลือก element ทั้งหมดที่มากกว่า 3
print(filtered_arr2d) # [4 5 6]

#  เปลี่ยนค่าเฉพาะ element ที่ตรงตามเงื่อนไข
arr[arr % 2 == 0] = 0  # เปลี่ยน element ที่เป็นเลขคู่เป็น 0
print(arr) # [1 0 3 0 5]
```

**6. Slicing**

Slicing ใช้ดึงส่วนของ array ออกมา โดยใช้รูปแบบ `array_variable[start_idx:stop_idx:step]`  `stop_idx` จะไม่ถูกรวมในผลลัพธ์  ค่า default ของ `start_idx` คือ 0, `stop_idx` คือขนาดของมิติ, และ `step` คือ 1. Slicing สร้าง view ของ array เดิม การเปลี่ยนแปลงค่าใน sliced array จะส่งผลต่อ original array ด้วย  ถ้าต้องการ copy  ต้องใช้ `.copy()`

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6])

print(arr[1:4]) # [2 3 4]
print(arr[:3])  # [1 2 3]
print(arr[3:])  # [4 5 6]
print(arr[::2]) # [1 3 5] (every other element)
print(arr[::-1]) # [6 5 4 3 2 1] (reverse array)

arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(arr2d[:2, :])  # แถวที่ 0 และ 1, ทุกคอลัมน์
# [[1 2 3]
#  [4 5 6]]

print(arr2d[:, 1:])  # ทุกแถว, คอลัมน์ที่ 1 และ 2
# [[2 3]
#  [5 6]
#  [8 9]]

#  Slicing สร้าง view
sliced_arr = arr[1:4]
sliced_arr[0] = 10
print(arr) # [ 1 10  3  4  5  6] (original array ก็เปลี่ยน)

# สร้าง copy โดยใช้ .copy()
sliced_copy = arr[1:4].copy()
sliced_copy[0] = 20
print(arr) # [ 1 10  3  4  5  6] (original array ไม่เปลี่ยน)

```

**7. Joining Arrays**

`np.concatenate()`, `np.stack()`, `np.hstack()`, `np.vstack()`:  ใช้เชื่อม array หลายๆ array เข้าด้วยกัน

```python
import numpy as np

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# np.concatenate():  เชื่อมตามแกนที่กำหนด
joined_arr = np.concatenate((arr1, arr2))
print("concatenate:", joined_arr) # [1 2 3 4 5 6]

arr3 = np.array([[1, 2], [3, 4]])
arr4 = np.array([[5, 6]])

joined_arr_rows = np.concatenate((arr3, arr4), axis=0) # เชื่อมตามแถว (axis=0)
print("concatenate axis=0:\n", joined_arr_rows)

joined_arr_cols = np.concatenate((arr3, arr4.T), axis=1) # เชื่อมตามคอลัมน์ (axis=1), .T คือ transpose
print("concatenate axis=1:\n", joined_arr_cols)



# np.stack(): สร้าง array ใหม่โดยเพิ่มมิติ
stacked_arr = np.stack((arr1, arr2), axis=0) #  สร้าง array 2 มิติ
print("stack axis=0:\n", stacked_arr)

stacked_arr = np.stack((arr1, arr2), axis=1) #  สร้าง array 2 มิติ
print("stack axis=1:\n", stacked_arr)

# np.hstack():  เชื่อมตามแนวนอน (horizontal)
hstacked_arr = np.hstack((arr3, np.array([[7],[8]])))
print("hstack:\n", hstacked_arr)

# np.vstack(): เชื่อมตามแนวตั้ง (vertical)
vstacked_arr = np.vstack((arr3, arr4))
print("vstack:\n", vstacked_arr)

```

**8. Splitting Arrays**

`np.split()`, `np.hsplit()`, `np.vsplit()`:  ใช้แบ่ง array ออกเป็น sub-arrays หลายๆ ส่วน

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6])

# np.split(): แบ่ง array ตาม index ที่กำหนด
split_arr = np.split(arr, 3) # แบ่งเป็น 3 ส่วนเท่าๆ กัน
print("split into 3 equal parts:", split_arr)

split_arr2 = np.split(arr, [2, 4]) # แบ่งตาม index ที่กำหนด (2 และ 4)
print("split at indices 2 and 4:", split_arr2)  

#  np.hsplit(): แบ่ง array ตามแนวนอน (horizontal)
arr2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
hsplit_arr = np.hsplit(arr2d, 2) # แบ่งเป็น 2 ส่วนเท่าๆ กัน
print("hsplit:\n", hsplit_arr)

#  np.vsplit(): แบ่ง array ตามแนวตั้ง (vertical)
vsplit_arr = np.vsplit(arr2d, 2) # แบ่งเป็น 2 ส่วนเท่าๆ กัน
print("vsplit:\n", vsplit_arr)
```

**9. Arithmetic Operators**

NumPy รองรับการคำนวณทางคณิตศาสตร์แบบ element-wise  หมายความว่า operator จะถูกนำไปใช้กับ element ที่ตำแหน่งเดียวกันในแต่ละ array.

```python
import numpy as np

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

print("Addition:", arr1 + arr2)  # [5 7 9]
print("Subtraction:", arr1 - arr2) # [-3 -3 -3]
print("Multiplication:", arr1 * arr2) # [ 4 10 18]
print("Division:", arr1 / arr2) # [0.25 0.4  0.5 ]
print("Power:", arr1 ** arr2) # [  1  32 729]
print("Modulo:", arr2 % arr1) # [0 1 0]



#  การคำนวณกับ scalar
print("Scalar Multiplication:", arr1 * 2) # [2 4 6]

#  NumPy functions
print("np.add():", np.add(arr1, arr2))
print("np.subtract():", np.subtract(arr1, arr2))
print("np.multiply():", np.multiply(arr1, arr2))
print("np.divide():", np.divide(arr1, arr2))
print("np.power():", np.power(arr1, arr2))

```

**10. Broadcasting**

Broadcasting เป็นกลไกสำคัญที่ทำให้ NumPy สามารถคำนวณ array ที่มี shape ต่างกันได้ โดย array ที่เล็กกว่าจะถูก "ขยาย" (ไม่ใช่การ copy จริงๆ) ให้มี shape ที่เข้ากันได้กับ array ที่ใหญ่กว่าก่อนทำการคำนวณ  มีกฎเกณฑ์ในการ broadcast ดังนี้:

1. เปรียบเทียบ shape ของ array ทั้งสองจากขวาไปซ้าย
2. มิติที่ขนาดเท่ากัน หรือ มิติใดมิติหนึ่งมีขนาดเป็น 1  สามารถ broadcast ได้
3. ถ้าขนาดของมิติไม่เท่ากัน และไม่มีมิติใดมีขนาดเป็น 1  จะเกิด error

```python
import numpy as np

#  ตัวอย่างที่ broadcast ได้
arr1 = np.array([1, 2, 3]) # shape (3,)
arr2 = 2 # shape ()  มองเหมือน (1,)

print("Broadcasting Example:", arr1 * arr2) # [2 4 6]  arr2 ถูก broadcast เป็น [2 2 2]

arr3 = np.array([[1, 2, 3], [4, 5, 6]]) # shape (2, 3)
arr4 = np.array([1, 2, 3]) # shape (3,)

print("Broadcasting Example 2D:\n", arr3 + arr4) # arr4 ถูก broadcast เป็น [[1, 2, 3], [1, 2, 3]]
# Output:
# [[2 4 6]
#  [5 7 9]]

#  ตัวอย่างที่ broadcast ไม่ได้
arr5 = np.array([[1, 2], [3, 4]]) # shape (2, 2)
arr6 = np.array([1, 2, 3]) # shape (3,)

print(arr5 + arr6) #  ValueError: operands could not be broadcast together with shapes (2,2) (3,)

```



**11. Arithmetic Operations (Methods) **

NumPy มี methods มากมายสำหรับคำนวณทางคณิตศาสตร์กับ array

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

print("Sum:", arr.sum()) # 15
print("Min:", arr.min()) # 1
print("Max:", arr.max()) # 5
print("Cumulative Sum:", arr.cumsum()) # [ 1  3  6 10 15]
print("Mean:", arr.mean()) # 3.0
print("Standard Deviation:", np.std(arr)) # 1.4142135623730951
print("Variance:", np.var(arr)) # 2.0

#  ใช้กับ array หลายมิติ  พร้อมระบุ axis
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print("Sum along rows (axis=0):", arr2d.sum(axis=0)) # [5 7 9]
print("Sum along columns (axis=1):", arr2d.sum(axis=1)) # [ 6 15]

print("np.median():", np.median(arr)) # 3.0  (มัธยฐาน)
print("np.percentile():", np.percentile(arr, 75)) # 4.0 (เปอร์เซ็นไทล์ที่ 75)

```

**12. Sorting**

`np.sort()`, `np.argsort()`: ใช้เรียงลำดับข้อมูลใน array. `np.sort()` คืนค่า array ที่เรียงลำดับแล้ว,  ส่วน `np.argsort()` คืนค่า indices ที่เรียงลำดับแล้ว

```python
import numpy as np

arr = np.array([3, 1, 4, 2, 5])

# np.sort():  คืนค่า array ที่เรียงลำดับแล้ว  (ไม่เปลี่ยน original array)
sorted_arr = np.sort(arr) 
print("Sorted Array:", sorted_arr) # [1 2 3 4 5]
print("Original Array:", arr) # [3 1 4 2 5] (ไม่เปลี่ยน)



# เรียงลำดับ inplace (เปลี่ยน original array)
arr.sort()
print("Sorted Array (inplace):", arr) # [1 2 3 4 5]

# np.argsort(): คืนค่า indices ที่เรียงลำดับแล้ว
indices = np.argsort(arr)
print("Sorted Indices:", indices) # [0 1 2 3 4] (ตอนนี้ arr เรียงแล้ว)

#  เรียงลำดับ array 2 มิติ
arr2d = np.array([[3, 1], [4, 2], [1, 5]])

sorted_arr2d_rows = np.sort(arr2d, axis=0) # เรียงตามคอลัมน์ (axis=0)
print("Sorted 2D Array (rows):\n", sorted_arr2d_rows)

sorted_arr2d_cols = np.sort(arr2d, axis=1) # เรียงตามแถว (axis=1)
print("Sorted 2D Array (columns):\n", sorted_arr2d_cols)



#  เรียงลำดับ structured array
data = np.array([('Alice', 25), ('Bob', 30), ('Charlie', 20)],
                 dtype=[('name', 'U10'), ('age', 'i4')])

sorted_data = np.sort(data, order='age')  # เรียงตาม field 'age'
print("Sorted Structured Array:\n", sorted_data)

```

**13. Searching**

`np.argmax()`, `np.argmin()`, `np.where()`, `np.searchsorted()`: ใช้ค้นหาข้อมูลใน array

```python
import numpy as np

arr = np.array([3, 1, 4, 2, 5])

# np.argmax():  คืนค่า index ของค่าสูงสุด
max_index = np.argmax(arr)
print("Index of Max Value:", max_index) # 4

# np.argmin():  คืนค่า index ของค่าต่ำสุด
min_index = np.argmin(arr)
print("Index of Min Value:", min_index) # 1

# np.where(): คืนค่า index ของ element ที่ตรงตามเงื่อนไข
indices = np.where(arr > 2)
print("Indices where value > 2:", indices)  # (array([0, 2, 4]),)

# np.searchsorted(): ค้นหาตำแหน่งที่ควรแทรกค่าลงใน array ที่เรียงลำดับแล้ว เพื่อให้ array ยังคงเรียงลำดับอยู่
arr_sorted = np.array([2, 4, 6, 8])
index = np.searchsorted(arr_sorted, 5)
print("Index to insert 5:", index) # 2

# ใช้กับ array หลายมิติ  พร้อมระบุ axis
arr2d = np.array([[3, 1, 5], [4, 2, 6]])
max_indices_rows = np.argmax(arr2d, axis=1) # หา index ของค่าสูงสุดในแต่ละแถว
print("Max indices along rows:", max_indices_rows) # [2 2]

max_indices_cols = np.argmax(arr2d, axis=0) # หา index ของค่าสูงสุดในแต่ละคอลัมน์
print("Max indices along cols:", max_indices_cols) # [1 0 1]

```


 หวังว่าสรุปฉบับละเอียดพร้อมคำอธิบายและตัวอย่างนี้จะเป็นประโยชน์นะครับ.
