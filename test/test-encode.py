chars = "测试"

# 编码与解码测试
for char in chars:
    c = ord(char) # 10进制的ASCII码
    o = chr(c)
    print(f"编码前{char}，编码后{c},反编码结果{o}")


# 十进制与二进制转换测试
print(int("11111111",2)) # 255


print(bin(255)) # 0b11111111
