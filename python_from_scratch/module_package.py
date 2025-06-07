import sys
sys.path.append('.')

from module_package import *
#from module_package.substract import substract
#from module_package.add import add


def main():
    a = add(1,3)
    b= substract(1,3)

    print(a,b)


if __name__ == '__main__':
    main()