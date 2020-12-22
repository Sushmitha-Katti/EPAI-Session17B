# write a python program to add two list of same length.
num1 = [1,2,3]
num2 = [4,5,6]
sum = num1 + num2
print(f'Sum: {sum}')

# write a python program Convert KM/H to MPH
kmh = 50
mph =  0.6214 * kmh
print("Speed:", kmh, "KM/H = ", mph, "MPH")

# write a program to find and print the smallest among three numbers
num1 = 100
num2 = 200
num3 = 300
if (num1 <= num2) and (num1 <= num3):
    smallest = num1
elif (num2 <= num1) and (num2 <= num3):
    smallest = num2
else:
    smallest = num3
print(f'smallest:{smallest}')

# write a function to sort a list
raw_list = [-5, -23, 5, 0, 23, -6, 23, 67]
sorted_list = []
while raw_list:
    minimum = raw_list[0]   
    for x in raw_list: 
        if x < minimum:
            minimum = x
    sorted_list.append(minimum)
    raw_list.remove(minimum)    

print(soreted_list)

# write a function to print the time it takes to run a function
import time
def time_it(fn, *args, repetitons= 1, **kwargs):
    start = time.perf_counter()
    if (repetitons <= 0):
        raise ValueError("repetitions should be greater that 0")
    if (not(isinstance(repetitons,int))):
        raise ValueError("Repetions must be of type Integer")
    for _ in range(repetitons):
        fn(*args, **kwargs)
    stop = time.perf_counter()
    return ((stop - start)/repetitons)



# write a python function to calculate simple Interest
def simple_interest(p,t,r): 
   
    si = (p * t * r)/100
    return si 

# write a python program to print all Prime numbers in an Interval
start = 11
end = 25
 
for i in range(start,end):
  if i>1:
    for j in range(2,i):
        if(i % j==0):
            break
    else:
        print(i)

# write a python funtion to implement a counter to record how many time the word has been repeated using closure concept
def word_counter():
    counter = {}
    def count(word):
        counter[word] = counter.get(word, 0) + 1
        return counter[word]
    return count

# write a  python program to check and print if a string is palindrome or not
st = 'malayalam'
j = -1
flag = 0
for i in st:
    if i != st[j]:
      j = j - 1
      flag = 1
      break
    j = j - 1
if flag == 1:
    print("Not a palindrome")
else:
    print("It is a palindrome")

# write a python function to find the URL from an input string using the regular expression
import re 
def Find(string): 
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex,string)       
    return [x[0] for x in url] 
      
# write a python program to find N largest elements from a list
l = [1000,298,3579,100,200,-45,900] 
n = 4
l.sort() 
print(l[-n:])

# write a python program to add two lists using map and lambda
nums1 = [1, 2, 3]
nums2 = [4, 5, 6]
result = map(lambda x, y: x + y, nums1, nums2)
print(list(result))

# write a python functionto test the equality of the float numbers
def float_equality_testing(a, b):
    
    rel_tol = 1e-12
    abs_tol = 1e-05
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

# write a python function to caclucate the polygon_area
def polygon_area( side_length, sides = 3):
    if(sides < 3 or sides > 6 ):
        raise ValueError("number of sides must be greater than 2 and less than 7")
    if(side_length < 0 ):
        raise ValueError("side length must be positive")

    return sides * (side_length ** 2) / (4 * tan(pi / sides))

# write a python program to get positive elements from given list of lists
Input = [[10, -11, 222], [42, -222, -412, 99, -87]] 
temp = map(lambda elem: filter(lambda a: a>0, elem), Input) 
Output = [[a for a in elem if a>0] for elem in temp] 

# write the program to remove empty tuples from a list
def Remove(tuples): 
    tuples = filter(None, tuples) 
    return tuples 
# write  a python program to find Cumulative sum of a list
list=[10,20,30,40,50]
new_list=[] 
j=0
for i in range(0,len(list)):
    j+=list[i]
    new_list.append(j) 
     
print(new_list) 
# write a python function to convert a list to string
s = ['I', 'want', 4, 'apples', 'and', 18, 'bananas'] 
listToStr = ' '.join(map(str, s)) 
print(listToStr)

# write a python program to merge 2 dictionaries
x = {'a' : 1, 'b' : 2, 'c' : 3}
y = {'x' : 10, 'y' : 20, 'z' : 30 }
z = {**x , **y}

# write a python code to implement Sigmoid function
import math
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# write a python code to implement RELU function
def relu(array):
    return [max(0,i) for i in array if(isinstance(i, int) or isinstance(i, float))]

# write a python function to check whether the given number is fibonacci or not
def fiboacci_number_check(n):
    if(isinstance(n,int)):
        result = list(filter(lambda num : int(math.sqrt(num)) * int(math.sqrt(num)) == num, [5*n*n + 4,5*n*n - 4] ))
        return bool(result) 
    else:
        raise TypeError("Input should be of type Int") 

# write a python program to strip all the vowels in a string
string = "Remove Vowel"
vowel = ['a', 'e', 'i', 'o', 'u']
"".join([i for i in string if i not in vowel]

# write a python program to give the next fibonacci number

    num_1, num_2,count = 0, 1,0

    def next_fibbonacci_number() :
    
        nonlocal num_1, num_2, count

        if(count == 0):
            count+=1
            return 0
        elif(count==1):
            count+=1
            return num_2
        else:
            num_1, num_2 = num_2, num_1+num_2
            return num_2

    return next_fibbonacci_number
# write a python function to calculate factorial of a given number
def factorial(n):
    fact = 1
    for num in range(2, n + 1):
        fact = fact * num
    return(fact)
# write a python program which will find all such numbers which are divisible by 7 but are not a multiple of 5 ;between 2000 and 3200 (both included)
l=[]
for i in range(2000, 3201):
    if (i%7==0) and (i%5!=0):
        l.append(str(i))

print(','.join(l))

# write the python program to generate a random number between 0 and 9 
import csv
def read_csv(input_file):
    with open(input_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
                print(f'{row}')
                break

# write a python program to Generate a Random Number
import random
print(random.randint(0,9))

# write a python program to Check Leap Year
year = 2000
if (year % 4) == 0:
   if (year % 100) == 0:
       if (year % 400) == 0:
           print(f"{year} is a leap year")
       else:
           print(f"{year} is not a leap year")
   else:
       print(f"{year} is a leap year")
else:
   print(f"{year} is not a leap year")

# write a python function to Compute LCM
def compute_lcm(x, y):
   if x > y:
       greater = x
   else:
       greater = y

   while(True):
       if((greater % x == 0) and (greater % y == 0)):
           lcm = greater
           break
       greater += 1

   return lcm
# write a python function to compute gcd
def compute_gcd(x, y):

   while(y):
       x, y = y, x % y
   return x

# write a python program to Remove Punctuations From a String
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
my_str = "Hello!!!, he said ---and went."
no_punct = ""
for char in my_str:
   if char not in punctuations:
       no_punct = no_punct + char
print(no_punct)

# write a python function to Find Hash of File
import hashlib
def hash_file(filename):

   h = hashlib.sha1()
   with open(filename,'rb') as file:
       chunk = 0
       while chunk != b'':
           chunk = file.read(1024)
           h.update(chunk)
   return h.hexdigest()
# write a python Program to Find the Size (Resolution) of a  JPEG Image and print it
def jpeg_res(filename):
   with open(filename,'rb') as img_file:
       img_file.seek(163)

       a = img_file.read(2)

       # calculate height
       height = (a[0] << 8) + a[1]

       # next 2 bytes is width
       a = img_file.read(2)

       # calculate width
       width = (a[0] << 8) + a[1]

   print("The resolution of the image is",width,"x",height)

# write a python program to count the number of each vowels
ip_str = 'Hello, have you tried our tutorial section yet?'
ip_str = ip_str.casefold()
count = {x:sum([1 for char in ip_str if char == x]) for x in 'aeiou'}
print(count)
        
# write a python Program to Find ASCII Value of Character
c = 'p'
print("The ASCII value of '" + c + "' is", ord(c))

# write a python Program to Solve Quadratic Equation
import cmath
a = 1
b = 5
c = 6
d = (b**2) - (4*a*c)
sol1 = (-b-cmath.sqrt(d))/(2*a)
sol2 = (-b+cmath.sqrt(d))/(2*a)
print('The solution are {0} and {1}'.format(sol1,sol2))

# write a python program to Convert Celsius To Fahrenheit
celsius = 37.5
fahrenheit = (celsius * 1.8) + 32
print(f'{celsius} degree Celsius is equal to {fahrenheit} degree Fahrenheit')

# write a python program to check Armstrong number of n digits
num = 1634
order = len(str(num))
sum = 0
temp = num
while temp > 0:
   digit = temp % 10
   sum += digit ** order
   temp //= 10
if num == sum:
   print(num,"is an Armstrong number")
else:
   print(num,"is not an Armstrong number")

# write a Python Program to Find the Sum of Natural Numbers
num = 16
if num < 0:
   print("Enter a positive number")
else:
   sum = 0
   while(num > 0):
       sum += num
       num -= 1
   print("The sum is", sum)

# write a python program  to Shuffle Deck of Cards
import itertools, random
deck = list(itertools.product(range(1,14),['Spade','Heart','Diamond','Club']))
random.shuffle(deck)
print(deck)

# write a Python function to Convert Decimal to Binary
def convertToBinary(n):
   if n > 1:
       convertToBinary(n//2)
   print(n % 2,end = '')

# wrtie a python function to solve Tower Of Hanoi and print necessary statements
def TowerOfHanoi(n , source, destination, auxiliary): 
    if n==1: 
        print("Move disk 1 from source",source,"to destination",destination) 
        return
    TowerOfHanoi(n-1, source, auxiliary, destination) 
    print("Move disk",n,"from source",source,"to destination",destination) 
    TowerOfHanoi(n-1, auxiliary, destination, source) 

# write a python function to find the number of times every day occurs in a Year and print them
import datetime  
import calendar 
   
def day_occur_time(year): 
    days = [ "Monday", "Tuesday", "Wednesday",   
           "Thursday",  "Friday", "Saturday",  
           "Sunday" ] 
    L = [52 for i in range(7)] 

    pos = -1
    day = datetime.datetime(year, month = 1, day = 1).strftime("%A") 
    for i in range(7): 
        if day == days[i]: 
            pos = i 
    if calendar.isleap(year): 
        L[pos] += 1
        L[(pos+1)%7] += 1       
    else: 
        L[pos] += 1

    for i in range(7): 
        print(days[i], L[i])

# write a python Program to Determine all Pythagorean Triplets in the Range
limit= 50
c=0
m=2
while(c<limit):
    for n in range(1,m+1):
        a=m*m-n*n
        b=2*m*n
        c=m*m+n*n
        if(c>limit):
            break
        if(a==0 or b==0 or c==0):
            break
        print(a,b,c)
    m=m+1

# function to Convert Binary to Gray Code
def binary_to_gray(n):
    n = int(n, 2) 
    n ^= (n >> 1)
    return bin(n)[2:]

# write a Python function to Find the Intersection of Two Lists
def intersection(a, b):
    return list(set(a) & set(b))

# write a python program to Remove the Given Key from a Dictionary
d = {'a':1,'b':2,'c':3,'d':4}
key= 'd'
if key in d: 
    del d[key]
else:
    print("Key not found!")
    exit(0)

# write a python function to Count the Number of Words in a Text File and print it
def word_count(fname) : 
    num_words = 0
    with open(fname, 'r') as f:
        for line in f:
            words = line.split()
            num_words += len(words)
    print(num_words)

# write a python function to Count Set Bits in a Number
def count_set_bits(n):
    count = 0
    while n:
        n &= n - 1
        count += 1
    return count

# wrie a python  Program to Flatten a List without using Recursion
a=[[1,[[2]],[[[3]]]],[[4],5]]
flatten=lambda l: sum(map(flatten,l),[]) if isinstance(l,list) else [l]
print(flatten(a))

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

