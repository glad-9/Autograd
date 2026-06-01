from scalar.value import Value

a = Value(2.0)
b = Value(3.0)
c = Value(1.0)

L = a * b

print(L._prev)
