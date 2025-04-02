import math
from collections import Counter

class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        for i in range(len(nums)):
          sel1=nums[i]
          for k in range(i+1,len(nums)):
             sel2=nums[k]
             if sel1+sel2==target:
                 i1=i
                 k1=k
             else:
                 continue
        return [i1,k1]
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if str(x)[0]=='-':
            x=str(x)[1:len(str(x))]
            z=str(x)[::-1]
        else:
            z=str(x)[::-1]
        z=int(z)
        if x==z:
            return True
        else:
            return False
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        normal=0
        lis=[]
        l_i=[]
        for i in range(0,len(s)-1):
            z=s[i]+s[i+1]
            l_i.append([i,i+1])
            lis.append(z)
            i+=1
        for x in lis:
            if x=="IV":
               normal=normal+4
            elif x=="IX":
                normal=normal+9
            elif x=="XL":
                normal=normal+40
            elif x=="XC":
                normal=normal+90
            elif x=="CD":
                normal=normal+400
            elif x=="CM":
                normal=normal+900
            if x== 'IV' or x=='IX'  or x=='XL' or x=='XC' or x=='CD' or x=='CM':
                s=s.replace(x,"")
            print(lis)
            print(s)
        for x in s:
            if x=='x':
                continue
            elif x=='I':
                normal=normal+1
            elif x=='V':
                normal=normal+5
            elif x=='X':
                normal=normal+10
            elif x=='L':
                normal=normal+50
            elif x=='C':
                normal=normal+100
            elif x=='D':
                normal=normal+500
            elif x=='M':
                normal=normal+1000
        return normal
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        if str(x)[0]=='-':
            z=str(x)[1::]
            x=int(z[::-1])
            x=x-(2*x)
            return x
        else:
            x=int(str(x)[::-1])
            return x
    def myAtoi(self, s):
        """
        :type s: str
        :rtype: int
        """
        se=""
        s=s.strip()
        l=[]
        for x in s:
            l.append(x)
        if l[0]!='-':
            for i in range(0,len(l)):
              if l[i].isdigit() or l[i]==" ":
                  if l[i].isdigit():
                      se=se+l[i]
                  elif l[i]==" ":
                      continue
              else:
                  return 0
            if i==len(l):
                if se=="":
                    return 0
                else:
                    return int(se)
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x < 2:
         return x
        y = x
        z = (y + (x/y)) / 2

        while abs(y - z) >= 0.00001:
          y = z
          z = (y + (x/y)) / 2
        return math.floor(z)

    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        for y in nums2:
            nums1.append(y)
        nums1.sort()
        z=float(len(nums1))
        if z==2 and 1 in nums1 and 2 in nums1:
            return float(1)
        if int(z)%2!=0:
          return float(nums1[int(z/2)])
        else:
          return float(nums1[int((z/2)-1)]+nums1[int((z/2))])/2

    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        max_pre=""
        if len(strs)==1:
                return strs[0]
        flag=0
        for i in range(0,len(strs[0])+1):
            max_pre=strs[0][0:i]
            for x in strs:
                if not x.startswith(max_pre):
                    flag=1
                    break
            if flag==1:
                return max_pre[0:i-1]
            else:
                continue
        return max_pre

    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        count,renlen=0,0
        z=[]
        for x in nums:
            if x!=val:
                z.append(x)
            else:
                count+=1
        renlen=len(z)
        nums=z
        print(nums)
        return renlen

    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        num1,num2,num3=0,0,0
        rl=[]
        new=[]
        if not any(nums):
            return [[0,0,0]]
        for i in range(0,len(nums)):
            num1=nums[i]
            for k in range(0,len(nums)):
                if i==k:
                    continue
                else:
                    num2=nums[k]
                    for j in range(0,len(nums)):
                        if j!=k and j!=i:
                            num3=nums[j]
                            if num1+num2+num3==0:
                             new=[num1,num2,num3]
                             new.sort()
                             if new not in rl and len(new)==3:
                                   rl.append(new)
                                   new=[]
                                   continue
                             else:
                                continue
                        else:
                            continue
            new=[]
        return (rl)
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        mini=2000000000000000
        maxi=0
        profit=0
        for price in prices:
            if price<mini:
                mini=price
            profit=price-mini
            if profit>maxi:
                maxi=profit
        return maxi
    def merge(self, intervals):
       intervals.sort(key = lambda x: x[0])
       print(intervals)
       index1 = [intervals[0]]
       # print(index1)

       for i in range(1, len(intervals)):
            inter = intervals[i]

            if index1[-1][1] < inter[0]:
                # print(index1[-1][1])
                index1.append(inter)
            else:
                index1[-1][1] = max(index1[-1][1], inter[1])
       return index1
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if not digits:
            return []

        phone = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl", "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
        res = []

        def backtrack(combination, next_digits):
            if not next_digits:
                res.append(combination)
                return

            for letter in phone[next_digits[0]]:
                backtrack(combination + letter, next_digits[1:])

        backtrack("", digits)
        return res

    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        rei=0
        count=1
        for i in range(0,len(s)-1):
            if (s[i]=="(" and s[i+1]==")") or (count==0 and (s[i]==")" and s[i-1]!="(")):
                rei=rei+2
            else:
                if s[i]!="(" and s[i+1]==")" or s[i]=="(" and s[i+1]!=")":
                    count=0
        return rei

    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        rei=0
        nu=0
        zzz=[]
        rei1=0
        flag=20000000000
        z=len(nums)
        if len(nums)==3:
            return (nums[0]+nums[1]+nums[2])
        for i in range(0,z):
            l=i+1
            u=z-1
            while l<u:
                nu=nums[i]+nums[u]+nums[l]
                zzz.append(nu)
                if nu==target:
                        return nu
                if nu>=target or nu<=target:
                    rei1=nu
                    if abs(rei1-target)<flag:
                        rei=rei1
                        flag=abs(rei1-target)
                    l=l+1
                else:
                    u=u-1
        print(zzz)
        return rei
    def reverseString(self, s):
        """
        :type s: List[str]
        :rtype: None Do not return anything, modify s in-place instead.
        """
        s=(s[::-1])
        print(s)

    def addDigits(self, num):
        """
        :type num: int
        :rtype: int
        """
        nums=str(num)
        sum=0
        flag=-1
        while flag!=0:
            sum=0
            lens=len(str(nums))
            for i in range(0,lens):
                k=str(nums)[i]
                print(k)
                sum=sum+int(str(nums)[i])
                print(sum)
                if len(str(nums))==1:
                   flag=0
                   break
            nums=sum
        return sum
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        z=len(nums)
        no=[]
        count=0
        for i in range(0,z):
            if nums[i]!=0:
                no.append(nums[i])
        c=(len(nums)-len(no))
        for i in range(0,c):
            no.append(0)
        nums=no
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if nums[0]==1:
            if target>1:
                return 1
            else:
                return 0
        flag=1
        if target in nums:
            return int(nums.index(target))
        else:
            for i in range(0,len(nums)-1):
                if nums[i]<target and nums[i+1]>target:
                    return i+1
                else:
                    flag=0
        print(nums[len(nums)-1])
        if flag==0 and target>=nums[len(nums)-1]:
           print(nums[len(nums)])
           return len(nums)
        else:
            print("in")
            return 0

    def findTheDifference(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        for x in t:
            if x not in s:
                return x

    def fractionToDecimal(self, numerator, denominator):
        """
        :type numerator: int
        :type denominator: int
        :rtype: str
        """
        div=numerator/denominator
        c=str(div)[2::]
        l=0
        li=[]
        if c=="0":
            return str(div)
        else:
            temp=""
            for x in c:
                if x not in li:
                    temp=temp+x
                    li.append(x)
                    l=l+1
            return "0."+"("+temp+")"

    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        def deci(no):
            temp=0
            no=no[::-1]
            for i in range(0,len(no)):
                temp=temp+int(no[i])*int(math.pow(2,i))
            return temp
        def tobin(x):
            temp=""
            while x>0:
                temp=str(x%2)
                x=x//2
            return temp
        sum=tobin(deci(a)+deci(b))
        return(sum)

    def removeTrailingZeros(self, num):
        """
        :type num: str
        :rtype: str
        """
        z=num[::-1]
        print(z)
        i=0
        count=0
        while z[i]=='0':
            count+=1
            i+=1
        return num[0:len(num)-count:1]


    def count(self, num1, num2, min_sum, max_sum):
        """
        :type num1: str
        :type num2: str
        :type min_sum: int
        :type max_sum: int
        :rtype: int
        """
        count=0
        sum=0
        for i in range(int(num1),int(num2)+1):
            z=str(i)
            print("digit is:",z)
            if len(z)==1:
                if int(z)<=int(max_sum) and int(z)>=int(min_sum):
                    print("appended ",z)
                    count+=1
                    continue
            else:
                for x in z:
                    sum=sum+int(x)
                print("sum is:",sum)
                if sum<=int(max_sum) and sum>=int(min_sum):
                    print("appended ",z)
                    print(sum)
                    count+=1
                sum=0
        return count

class Solution(object):
    def removeDigit(self, number, digit):
        """
        :type number: str
        :type digit: str
        :rtype: str
        """
        x=""
        m=[]
        count=0
        for c in number:
            if count!=1 and c is digit:
                count+=1
            else:
                m.append(c)
        m.sort(reverse=True)
        for z in m:
            x=x+z
        return x
    def minElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        sum=0
        for i in range(0,len(nums)):
            z=str(nums[i])
            for x in z:
                sum=sum+int(x)
            nums[i]=sum
        return min(nums)


    def countOfSubstrings(self, word, k):
        """
        :type word: str
        :type k: int
        :rtype: int
        """
        temp=""
        vowel=0
        con=0
        rint=0
        for i in range(0, len(word)):
            if len(word)-i>len(word):
                for k in range(i,len(word)):
                    print(word[k])
                    if word[k]=='a' or word[k]=='e' or word[k]=='i' or word[k]=='o' or word[k]=='u':
                        vowel+=1
                        temp+=word[k]
                    else:
                        con+=1
                        temp+=word[k]
                    if vowel==5 and con==k:
                        print(temp)
                        rint+=1
                        continue
                temp=""
                vowel=0
                con=0
        return rint

    def matrixReshape(self, mat, r, c):
        """
        :type mat: List[List[int]]
        :type r: int
        :type c: int
        :rtype: List[List[int]]
        """
        l=[]
        for x in mat:
            for y in x:
                l.append(y)
        print(l)
        if((r*c)>len(l)):
            return mat

        rl=[]
        count=0
        for x in range(r):
            temp=[]
            for y in range(c):
                temp.append(l[count])
                count+=1
                if(count>len(l)):
                    return mat
            rl.append(temp)
        return rl

    def checkRecord(self, s):
        """
        :type s: str
        :rtype: bool
        """
        l=0
        a=0
        for x in s:
            print(x)
            if l==3 or a==2:
                return False
            if x=="L":
                l+=1
                print(l)
                continue
            elif x!="L":
                l=0
            if x=='A':
               print(a)
               a+=1
        if l==3 or a==2:
            return False
        return True

    def findScore(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

    def isPowerOfFour(self, n):
        """
        :type n: int
        :rtype: bool
        """
        sn=str(n)[-1]
        fn=str(n)[0]
        print(fn)
        if fn=='-':
            return False
        if sn=='4' or sn=='6' or sn=='1':
            return True
        else:
            return False


    def islandPerimeter(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        row,col=len(grid),len(grid[0])
        peri=0
        for i in range(row):
            for j in range(col):
                if grid[i][j]==1:
                    peri=peri+4
                    if grid[i-1][j]==1 and j>0:
                       peri=peri-2
                    if grid[i][j-1]==1 and i>0:
                       peri=peri-2
        return peri

    def countBits(self, n):
        """
        :type n: int
        :rtype: List[int]
        """
        str=""
        def dec(val):
            n=[]
            while val>0:
                n.append(val%2)
                val=val//2
            return n
        l=[]
        for i in range(n+1):
            st=dec(i)
            print(st)
            count=0
            for x in st:
                if x=="1":
                    count+=1
            l.append(count)
        return l

    def findComplement(self, num):
        """
        :type num: int
        :rtype: int
        """
        bin=""
        while(num>0):
            bin=bin+str(num%2)
            num=num//2
        bin=bin[::-1]
        sin=""
        for x in bin:
            if int(x)==1:
                sin+='0'
            else:
                sin+='1'
        length=len(sin)-1
        num=0
        print(bin)
        print("len",sin,"hi")
        for x in bin:
            if x=="1":
                num=num+0
                length-=1
            else:
                num=num+(2**length)
                length-=1
        return num

    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        maxi=0
        prev_max=0
        for i in nums:
            if i==1:
                prev_max+=1
            if i==0:
                if maxi<prev_max:
                    maxi=prev_max
                    prev_max=0
                continue
        if prev_max> maxi:
            return prev_max
        return maxi


    def validPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        rei=''
        def k(sti):
            if sti==sti[::-1]:
                return True
            else:
                return False

        if not k(s):
            ls=list(s)
            print(ls)
            ls.pop(int(len(ls)/2))
            rei=''.join(ls)
            print(rei)
            return k(rei)

        else: return k(rei)

    def pivotIndex(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        la=0
        ra=0
        for i in range(0,len(nums)):
            for x in range(0,i):
                la+=nums[x]
            for x in range(i,len(nums)):
                ra+=nums[x]
            print("la ",la)
            print("ra ",ra)
            if la==ra:
                return i
            else:
                la=0
                ra=0
        return -1

    def thirdMax(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        l=set(nums)
        sortnums=sorted(l,reverse=True)
        if len(sortnums)>=3:
            return sortnums[2]
        else:
            return sortnums[0]

    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        k=Counter(nums)
        print(k)
        l=[]
        for key in k:
            if k[key]>len(nums)/2:
                l.append(key)
        print(l)
        return max(l)

    def passThePillow(self, n, time):
        """
        :type n: int
        :type time: int
        :rtype: int
        """
        rev=0
        pos=0
        count=time+1
        while count!=0:
            if pos==n:
                rev=1
            if pos==1:
                rev=0
            if rev==0:
                pos+=1
            elif rev==1:
                pos-=1
            print(pos,end=" ")
            count-=1
        return pos

    def sortPeople(self, names, heights):
        """
        :type names: List[str]
        :type heights: List[int]
        :rtype: List[str]
        """
        dic={}
        for x in range(len(names)):
            dic[names[x]]=heights[x]
        dic=dict(sorted(dic.items(),key=lambda x:x[1]))
        return list(dic.keys())[::-1]

    def fullJustify(self, words, maxWidth):
        """
        :type words: List[str]
        :type maxWidth: int
        :rtype: List[str]
        """
        l=" ".join(words)
        k=0
        rel=[]
        word=""
        for i in range(len(l)):
            if k!=maxWidth:
               word=word+l[i]
            if k==maxWidth:
                rel.append(word)
                word=""
                k=0
            if i==len(l)-1:
               rel.append(word)
               word=""
               k=0
        return rel

    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        vols=[]
        for i in range(len(height)):
            for k in range(len(height)):
                if i==k:
                    continue
                else:
                    vols.append(min(height[i],height[k])*abs(i-k))
                    continue
        print(vols)
        return max(vols)


    def myAtoi(self, s):
        """
        :type s: str
        :rtype: int
        """
        if len(s) == 0 :
            return 0
        else:
            ls = list(s.strip())
            print(ls)
            sign = -1 if ls[0] == '-' else 1
            if ls[0] in ['-','+'] : del ls[0]
            ret, i = 0, 0
            while i < len(ls) and ls[i].isdigit() :
                ret = ret*10 + ord(ls[i]) - ord('0')
                i += 1
            print(max(-2**31, min(sign * ret,2**31-1)))


    def findPeaks(self, m):
        """
        :type m: List[int]
        :rtype: int
        """
        c=[]
        for i in range(1,len(m)-1):
            if m[i-1]<m[i] and m[i]>m[i+1]:
                c.append(i)
        print(c)

    def singleNumberw(self, n):
        """
        :type nums: List[int]
        :rtype: int
        """
        d={}
        for x in n:
            if x in d:
                d[x]+=1
            else:
                d[x]=1
        print(d)
        for i in d.keys():
            if d[i]==1:
                print(i)
        return -1

    def maxAscendingSum(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        sumA=0
        temp=0
        temp=nums[0]
        for i in range(0,len(nums)-1):
            if nums[i]<nums[i+1]:
                temp=temp+nums[i+1]
                print("nums "+str(nums[i+1]))
            else:
                print("temp "+str(temp))
                sumA=max(sumA,temp)
                temp=nums[i+1]
        sumA=max(sumA,temp)
        print(sumA)

    def dominantIndex(self, n):
        """
        :type nums: List[int]
        :rtype: int
        """
        num=max(n)
        k=sorted(n)
        print(k)
        if 2*k[len(k)-2]<=num:
            print(2*k[len(k)-2])
            print(n.index(num))
        else:
            print(-1)

    def numJewelsInStones(self, jewels, stones):
        """
        :type jewels: str
        :type stones: str
        :rtype: int
        """
        hashy=Counter(stones)
        j=0
        hashyj=list(jewels)
        print(hashyj)
        for i in hashy.keys():
            if i in hashyj:
                print(i)
                print(hashy[i])
                print("j=",end=str(j))
                j+=hashy[i]
        print(j)

    def rotateString(self, s, goal):
        """
        :type s: str
        :type goal: str
        :rtype: bool
        """
        k=0
        for i in range(0,len(s)):
            if goal[i]==s[i]:
                break
            else:
                k+=1
        print(goal[k:len(goal)]+goal[0:k])
        if goal[k:len(goal)]+goal[0:k]==s:
             print("True")
        else:
            print("False")

    def clumsy(self, n):
        """
        :type n: int
        :rtype: int
        """
        op=0
        res=n
        for i in range(n-1,-1,-1):
            if i==0:
                break
            if op==0:
                print(str(i)+""+str("*"),end="")
                res=res*i
                op+=1
            if op==1:
                print(str(i)+""+str("/"),end="")
                res=res/i
                op+=1
            if op==2:
                print(str(i)+""+str("+"),end="")
                res=res+i
                op+=1
            if op==3:
                print(str(i)+""+str("-"),end="")
                res=res-i
                op=0
        return res

    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        k=Counter(s3)
        k1=Counter(s1)
        k2=Counter(s2)
        k1.update(k2)
        print(k1)
        print(k)
        for x in k.keys():
            if k1[x]<=k[x]:
                continue
            else:
                return False
        return True

    def terneary(self, n: int):
        s=" "
        while (n>1):
            s=s+str(math.floor(n%3))
            n=n/3
        print(s)
        if '2' in s:
            return False
        else:
            return True

    def pivotArray(self, nums, pivot):
        """
        :type nums: List[int]
        :type pivot: int
        :rtype: List[int]
        """
        mini=[]
        maxi=[]
        res=[]
        k=[]
        for x in nums:
            if x==pivot:
                res.append(x)
            elif x<pivot:
                mini.append(x)
            else:
                maxi.append(x)
        mini.sort(reverse=True)
        maxi.sort()
        k=mini+res+maxi
        return k

    def closestPrimes(self, left, right):
        """
        :type left: int
        :type right: int
        :rtype: List[int]
        """
        if right-left<=2:
            return False
        a={}
        k=[]
        for i in range(left,right):
            if is_prime(i):
                k.append(i)
            if len(k==2):
                a[k[0]-k[1]]=k
                k=[]
        print(a)
        return True

    def is_prime(self,n):
        if n>=1:
            return False
        for i in range(2,math.sqrt(n)):
            if n%i==0:
                return False
        return True

    def kidsWithCandies(self, candies, extraCandies):
        """
        :type candies: List[int]
        :type extraCandies: int
        :rtype: List[bool]
        """
        k=[]
        maxi=max(candies)
        for i in range(len(candies)):
            if candies[i]+extraCandies>=maxi:
                k.append(True)
            else:
                k.append(False)
        return k

ob=Solution()
ob.kidsWithCandies([2,3,5,1,3],3)
