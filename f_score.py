b = 0.5
p = float(input("precision: "))
r = float(input("recall: "))

f_score = (1 + (b**2)) * ((p * r)/(((b**2) * p) + r))

print("F0.5: %s" % f_score)

