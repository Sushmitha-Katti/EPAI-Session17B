# write a python program to add two list of same length.
def add_two_list_items():
    num1 = [1,2,3]
    num2 = [4,5,6]
    sum = num1 + num2
    print(f'Sum: {sum}')


# write a python program to add numbers from two list if first list item is even and second list item is odd.
def add_two_lists_even_odd(l1, l2):
    new = []
    for x, y in zip(l1, l2):
        if l1%2 == 0 and l2%2 != 0:
            new.append(x+y)
    return new

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

import os
import nltk
import string
from collections import Counter
from itertools import permutations, combinations, combinations_with_replacement

letters = string.ascii_lowercase


# write a python function to print pyramid pattern
def pyramid_pattern(symbol='*', count=4):
    for i in range(1, count + 1):
        print(' ' * (count - i) + symbol * i, end='')
        print(symbol * (i - 1) + ' ' * (count - i))


# write a python function to count the occurrence of a given word in a given file
def check_word_count(word, file):
    if not os.path.isfile(file):
        raise FileNotFoundError
    if not isinstance(word, str):
        raise TypeError

    with open(file, 'r') as f:
        lines = f.readlines()
        words = [l.strip().split(' ') for l in lines]
        words = [word for sublist in words for word in sublist]
        c = Counter(words)
    return c.get(word, 0)


# write a python function to make permutations from a list with given length
def get_permutations(data_list, l=2):
    return list(permutations(data_list, r=l))


# write a python program to get all possible permutations of size of the string in lexicographic sorted order.
def get_ordered_permutations(word, k):
    [print(''.join(x)) for x in sorted(list(permutations(word, int(k))))]


# write a python program to get all possible combinations, up to size of the string in lexicographic sorted order.
def get_ordered_combinations(string, k):
    [print(''.join(x)) for i in range(1, int(k) + 1) for x in combinations(sorted(string), i)]


# write a python function to get all possible size replacement combinations of the string in lexicographic sorted order.
def get_ordered_combinations_with_replacement(string, k):
    [print(''.join(x)) for x in combinations_with_replacement(sorted(string), int(k))]



# write a python function for Caesar Cipher, with given shift value and return the modified text
def caesar_cipher(text, shift=1):
    alphabet = string.ascii_lowercase
    shifted_alphabet = alphabet[shift:] + alphabet[:shift]
    table = str.maketrans(alphabet, shifted_alphabet)
    return text.translate(table)


# write a python function for a string to swap the case of all letters.
def swap_case(s):
    return ''.join(x for x in (i.lower() if i.isupper() else i.upper() for i in s))


# write a python function to get symmetric difference between two sets from user.
def symmetric_diff_sets():
    M, m = input(), set(list(map(int, input().split())))
    N, n = input(), set(list(map(int, input().split())))
    s = sorted(list(m.difference(n)) + list(n.difference(m)))
    for i in s:
        print(i)


# write a python function to check if given set is subset or not
def check_subset():
    for _ in range(int(input())):
        x, a, z, b = input(), set(input().split()), input(), set(input().split())
    print(a.issubset(b))


# write a python program for basic HTML parser
from html.parser import HTMLParser


class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        for attr in attrs:
            print("->", attr[0], ">", attr[1])


parser = MyHTMLParser()

for i in range(int(input())):
    parser.feed(input())


# write a python function for Named Entity Recognizer using NLTK
def ner_checker(texts):
    all_set = set()

    def nltk_ner_check(texts):
        for i, text in texts:
            for entity in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text))):
                if isinstance(entity, nltk.tree.Tree):
                    etext = " ".join([word for word, tag in entity.leaves()])
                    # label = entity.label()
                    all_set.add(etext)

    nltk_ner_check(texts=texts)
    return all_set


#write a function to compress a given string. Suppose a character 'c' occurs consecutively X times in the string. Replace these consecutive occurrences of the character 'c' with  (X, c) in the string.
def compress(text):
    from itertools import groupby
    for k, g in groupby(text):
        print("({}, {})".format(len(list(g)), k), end=" ")


# write a python function to count 'a's in the repetition of a given string 'n' times.
def repeated_string(s, n):
    return s.count('a') * (n // len(s)) + s[:n % len(s)].count('a')


# write a python function to find all the substrings of given string that contains 2 or more vowels. Also, these substrings must lie in between 2 consonants and should contain vowels only.
def find_substr():
    import re
    v = "aeiou"
    c = "qwrtypsdfghjklzxcvbnm"
    m = re.findall(r"(?<=[%s])([%s]{2,})[%s]" % (c, v, c), input(), flags=re.I)
    print('\n'.join(m or ['-1']))


# write a python function that given five positive integers and find the minimum and maximum values that can be calculated by summing exactly four of the five integers.
def min_max():
    nums = [int(x) for x in input().strip().split(' ')]
    print(sum(nums) - max(nums), sum(nums) - min(nums))


# write a python function to find the number of (i, j) pairs where i<j and ar[i]+ar[j] is divisible by k in a data list
def divisible_sum_pairs(arr, k):
    count = 0
    n = len(arr)
    for i in range(n - 1):
        j = i + 1
        while j < n:
            if ((arr[i] + arr[j]) % k) == 0:
                count += 1
            j += 1
    return count


import math


# Write a python Class to calculate area of a circle and print the vale for a radius
class CircleArea:
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return math.pi * self.radius * self.radius


r = 2
obj = CircleArea(r)
print("Area of circle:", obj.area())


# Write a python function to count the number of Words in a Text File
def check_words():
    fname = input("file name: ")
    num_words = 0
    with open(fname, 'r') as f:
        for line in f:
            words = line.split()
            num_words += len(words)
    print("Number of words = ", num_words)


# Write a python function to Count the Number of Lines in a Text File
def check_lines():
    fname = input("file name: ")
    num_lines = 0
    with open(fname, 'r') as f:
        for line in f:
            num_lines += 1
    print("Number of lines = ", num_lines)


# Write a python function that Counts the Number of Blank Spaces in a Text File
def count_blank_space():
    fname = input("file name:")
    count = 0
    with open(fname, 'r') as f:
        for line in f:
            count += line.count(' ')
    return count


# Write a python function to check if 2 strings are anagrams or not
def anagram(s1, s2):
    if sorted(s1) == sorted(s2):
        return True
    else:
        return False


# Write a python function to remove the duplicate items from a List and return the modified data list
def remove_duplicates(data):
    c = Counter(data)
    s = set(data)
    for item in s:
        count = c.get(item)
        while count > 1:
            data.pop(item)
            count -= 1
    return data


# write a python function to get the most common word in text
def most_common(text):
    c = Counter(text)
    return c.most_common(1)


# write a python function to do bitwise multiplication on a given bin number by given shifts
def bit_mul(n, shift):
    return n << shift


# write a python function for bitwise division with given number of shifts
def bit_div(n, shift):
    return n >> shift


# write a python program to implement Queue
from collections import deque

class Queue():
    '''
    Thread-safe, memory-efficient, maximally-sized queue supporting queueing and
    dequeueing in worst-case O(1) time.
    '''


    def __init__(self, max_size = 10):
        '''
        Initialize this queue to the empty queue.

        Parameters
        ----------
        max_size : int
            Maximum number of items contained in this queue. Defaults to 10.
        '''

        self._queue = deque(maxlen=max_size)


    def enqueue(self, item):
        '''
        Queues the passed item (i.e., pushes this item onto the tail of this
        queue).

        If this queue is already full, the item at the head of this queue
        is silently removed from this queue *before* the passed item is
        queued.
        '''

        self._queue.append(item)


    def dequeue(self):
        '''
        Dequeues (i.e., removes) the item at the head of this queue *and*
        returns this item.

        Raises
        ----------
        IndexError
            If this queue is empty.
        '''

        return self._queue.pop()


# write a python function to get dot product between two lists of numbers
def dot_product(a, b):
    return sum(e[0] * e[1] for e in zip(a, b))


# write a python function to strip punctuations from a given string
def strip_punctuations(s):
    return s.translate(str.maketrans('', '', string.punctuation))


# write a python function that returns biggest character in a string
from functools import reduce


def biggest_char(string):
    if not isinstance(string, str):
        raise TypeError
    return reduce(lambda x, y: x if ord(x) > ord(y) else y, string)


# write a python function to Count the Number of Digits in a Number
def count_digits():
    n = int(input("Enter number:"))
    count = 0
    while n > 0:
        count = count + 1
        n = n // 10
    return count


# write a python function to count number of vowels in a string
def count_vowels(text):
    v = set('aeiou')
    for i in v:
        print(f'\n {i} occurs {text.count(i)} times')


# write a python function to check external IP address
def check_ip():
    import re
    import urllib.request as ur
    url = "http://checkip.dyndns.org"
    with ur.urlopen(url) as u:
        s = str(u.read())
        ip = re.findall(r"\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}", s)
        print("IP Address: ", ip[0])
        return ip[0]


# write a python function for some weird hypnosis text.
def weird():
    import random

    def getlength(script):
        return sum((i['length'] for i in script))

    def truncate(target_length, script):
        if getlength(script) > target_length:
            script = sorted(script, key=lambda k: (k['priority'], -k['length']))[:-1]
            return truncate(target_length, script)
        return sorted(script, key=lambda k: k['index'])

    def as_text(script):
        return "\n".join([i['text'] for i in script])

    priorities_and_sentences = [
        (1, "...now... sitting comfortably in the chair"),
        (2, "...with your feet still flat on the ground"),
        (3, "...back straight and head up right"),
        (2, "...make these adjustments now if you need to"),
        (3, "... pause.............................."),
        (1, "...your eyes ...still ...comfortably closed"),
        (2, "...nice and relaxed...comfortable and relaxed..."),
        (3, "... pause......................................."),
        (1, "...now...I want you to notice...how heavy your head is starting to feel..."),
        (1, "how heavy your head feels..."),
        (3, "... pause......................................."),
        (2, "really noticing the weight... of your head..."),
        (3,
         "and how much more ...comfortable...it will feel when you let your neck relaxes ...and your head begins to fall forward ...into a much more comfortable"),
    ]

    scriptlist = [{'priority': j[0], 'text': j[1], 'length': len(j[1]), 'index': i} for i, j in
                  enumerate(priorities_and_sentences)]

    print(as_text(truncate(500, scriptlist)))
    print(as_text(truncate(300, scriptlist)))
    print(as_text(truncate(200, scriptlist)))


# write a python function for dice roll asking user for input to continue and randomly give an output.
def dice():
    import random
    min = 1
    max = 6
    roll_again = 'y'

    while roll_again == "yes" or roll_again == "y":
        print("Rolling the dice...")
        print(random.randint(min, max))
        roll_again = input("Roll the dices again?")


from cryptography.fernet import Fernet


# write a python program to Encrypt and Decrypt features within 'Secure' class with key generation, using cryptography module
class Secure:
    def __init__(self):
        """
           Generates a key and save it into a file
        """
        key = Fernet.generate_key()
        with open("secret.key", "wb") as key_file:
            key_file.write(key)

    @staticmethod
    def load_key():
        """
        Load the previously generated key
        """
        return open("secret.key", "rb").read()

    def encrypt_message(self, message):
        """
        Encrypts a message
        """
        key = self.load_key()
        encoded_message = message.encode()
        f = Fernet(key)
        encrypted_message = f.encrypt(encoded_message)
        print("\nMessage has been encrypted: ", encrypted_message)
        return encrypted_message

    def decrypt_message(self, encrypted_message):
        """
        Decrypts an encrypted message
        """
        key = self.load_key()
        f = Fernet(key)
        decrypted_message = f.decrypt(encrypted_message)
        print("\nDecrypted message:", decrypted_message.decode())


s = Secure()
encrypted = s.encrypt_message("My deepest secret!")
s.decrypt_message(encrypted)


# write a python function to generate SHA256 for given text
def get_sha256(text):
    import hashlib
    return hashlib.sha256(text).hexdigest()


# write a python function to check if SHA256 hashed value is valid for given data or not
def check_sha256_hash(hashed, data):
    import hashlib
    return True if hashed == hashlib.sha256(data.encode()).hexdigest() else False


# write a python function to get HTML code for a given URL
def get_html(url="http://www.python.org"):
    import urllib.request

    fp = urllib.request.urlopen(url)
    mybytes = fp.read()
    mystr = mybytes.decode("utf8")
    fp.close()
    print(mystr)


# write a python function to get Bitcoin prices after every given 'interval' seconds
def get_btc_price(interval=5):
    import requests
    import json
    from time import sleep

    def getBitcoinPrice():
        URL = "https://www.bitstamp.net/api/ticker/"
        try:
            r = requests.get(URL)
            priceFloat = float(json.loads(r.text)["last"])
            return priceFloat
        except requests.ConnectionError:
            print("Error querying Bitstamp API")

    while True:
        print("Bitstamp last price: US $ " + str(getBitcoinPrice()) + "/BTC")
        sleep(interval)


# write a python function to get stock prices for a company from 2015 to 2020-12
def get_stock_prices(tickerSymbol='TSLA'):
    import yfinance as yf

    # get data on this ticker
    tickerData = yf.Ticker(tickerSymbol)

    # get the historical prices for this ticker
    tickerDf = tickerData.history(period='1d', start='2015-1-1', end='2020-12-20')

    # see your data
    print(tickerDf)


# write a python function to get 10 best Artists playing on Apple iTunes
def get_artists():
    import requests
    url = 'https://itunes.apple.com/us/rss/topsongs/limit=10/json'
    response = requests.get(url)
    data = response.json()
    for artist_dict in data['feed']['entry']:
        artist_name = artist_dict['im:artist']['label']
        print(artist_name)


# write a python function to get prominent words from user test corpus using TFIDF vectorizer
def get_words(corpus, new_doc, top=2):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer(stop_words='english')
    if not corpus:
        corpus = [
            'I would like to check this document',
            'How about one more document',
            'Aim is to capture the key words from the corpus',
            'frequency of words in a document is called term frequency'
        ]

    X = tfidf.fit_transform(corpus)
    feature_names = np.array(tfidf.get_feature_names())

    if not new_doc:
        new_doc = ['can key words in this new document be identified?',
                   'idf is the inverse document frequency calculated for each of the words']
    responses = tfidf.transform(new_doc)

    def get_top_tf_idf_words(response, top_n=top):
        sorted_nzs = np.argsort(response.data)[:-(top_n + 1):-1]
        return feature_names[response.indices[sorted_nzs]]

    print([get_top_tf_idf_words(response, 2) for response in responses])


# write a python function to generate wordcloud on given text or file
import os


def get_word(data):
    if not (isinstance(data, str) or os.path.isfile(data)):
        raise TypeError("Text must be string or a File object.")
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt
    stopwords = set(STOPWORDS)
    if os.path.isfile(data):
        with open(data, 'r') as f:
            data = f.read()

    data = ' '.join(data.lower().split(' '))
    wordcloud = WordCloud(width=400, height=400,
                          background_color='white',
                          stopwords=stopwords,
                          min_font_size=15).generate(data)

    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.show()


# get_word(data="./christmas_carol.txt")


# write a python function to sort each item in a data structure on one of the keys
def sort_list_with_key():
    animals = [
        {'type': 'lion', 'name': 'Mr. T', 'age': 7},
        {'type': 'tiger', 'name': 'scarface', 'age': 3},
        {'type': 'puma', 'name': 'Joe', 'age': 4}]
    print(sorted(animals, key=lambda animal: -animal['age']))


# write a python function with generator for an infinite sequence
def infinite_sequence():
    n = 0
    while True:
        yield n
        n += 1


import uuid


# write a python function to generate a Unique identifier across space and time in this cosmos.
def get_uuid():
    return uuid.uuid4()


import secrets


# write a python function to generate cryptographically strong pseudo-random data
def get_cryptographically_secure_data(n=101):
    return secrets.token_bytes(n), secrets.token_hex(n)


# write a python function to convert byte to UTF-8
def byte_to_utf8(data):
    return data.decode("utf-8")
print(byte_to_utf8(data=b'r\xc3\xa9sum\xc3\xa9'))




