# write a python program to add two numbers 
num1 = 10.5
num2 = 20.5
sum = num1 + num2
print(f'Sum: {sum}')

# write a python function to add two user provided numbers and return the sum
def add_two_numbers(num1, num2):
    sum = num1 + num2
    return sum

# write a program to find and print the largest among three numbers
num1 = 10
num2 = 12
num3 = 14
if (num1 >= num2) and (num1 >= num3):
    largest = num1
elif (num2 >= num1) and (num2 >= num3):
    largest = num2
else:
    largest = num3
print(f'largest:{largest}')

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

print soreted_list

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
# Python program to check and print if a string is palindrome or not
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

# Python code to find the URL from an input string using the regular expression
import re 
def Find(string): 
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex,string)       
    return [x[0] for x in url] 
      
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

