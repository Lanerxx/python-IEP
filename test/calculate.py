c1 = [ 13,2,7,15,10,11,14,16,17,18,1,3,6,8,5,4,9,12 ]
c2 = [ 6,7,5,16,13,14,17,10,3,15,1,9,18,11,8,4,12,2 ]
c3 = [ 3,14,13,10,5,15,8,9,16,6,1,12,11,2,4,7,18,17 ]
c4 = [ 6,10,9,11,15,12,13,18,5,4,2,8,7,1,3,17,16,14 ]

sum = [ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 ]

count = len(c1)
print(count)
for i in range(0, count):
    indexSum = c1[i] -1
    sum[indexSum]  = sum[indexSum] + (18-i)
    indexSum = c2[ i ] -1
    sum[ indexSum ] = sum[ indexSum ] + (18 - i)
    indexSum = c3[ i ] -1
    sum[ indexSum ] = sum[ indexSum ] + (18 - i)
    indexSum = c4[ i ] -1
    sum[ indexSum ] = sum[ indexSum ] + (18 - i)

for i in range(0, count):
    print(sum[i])