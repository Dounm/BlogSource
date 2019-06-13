---
title: "Scala语法简介"
date: 2016-09-10T23:46:08+08:00
tags: [Scala, Spark]
categories: [DistributedComputing]
toc: true
---


## 1. Scala语言特性

Scala(short for Scalable Language)是一种混合了面向对象和函数式编程的语言。

具有如下特性：

- 面向对象：Scala是一种纯的面向对象语言，每一个value都是对象
- 函数式：Scala也支持函数式编程，每一个函数都是value。并且支持很多函数式编程的特性：
    - anonymous function
    - higher-order function
    - nested functions
    - currying等等
- Scala是静态类型语言，使用type inference来推导变量类型而无需显示说明类型
- Scala跑在JVM上，并且能执行Java代码，使用Java库，语法和Java也很类似

## 2. Scala基本语法

### 2.1. 注释
    //单行注释
    /*多行注释*/
    
### 2.2. 分号
行末无需分号。但是对于多个语句在同一行时，需要分号`;`隔离

### 2.3. Package
同java

    package com.xxx.xxx
    import com.xxx.xxx
    
### 2.4. if-else
同java。但有时可以压缩为一个表达式

    if(condition) res1 else res2
    
### 2.5. Loop

for loop：

    /*for loop execution with range*/
    for( a <- 1 to 3; b < 1 until 3){
        println( "Value of a: " + a );
        println( "Value of b: " + b );
    }
    //1 until 3 not include 3
    //1 to 3 include 3
    //the for loop will iterate all possible computations
    
    /* for loop execution with a collection*/
    val numList = List(1,2,3,4,5,6);
    for( a <- numList ){
        println( "Value of a: " + a
    );

while loop: 同java
do-while loop: 同java

### 2.6. 变量
分为**可变变量**和**不可变变量**。

    var myVar : String = "Foo"      //mutable variable
    myVar = "Bar"
    
    val myVal : String = "Foo"      //immutable variable
    
    val myVal2 = "Foo"              //type inference
    
    /*multiple assignment*/
    val (myVar1: Int, myVar2: String) = Pair(40, "Foo")     
    val (myVar1, myVar2) = Pair(40, "Foo")
    
### 2.7. Class

    /*class without constructor*/
    class SimpleGreeter {
      val greeting = "Hello, world!"
      def greet() = println(greeting)
    }
    val g = new SimpleGreeter
    g.greet()
     
    /*class with primary constructor*/
    //类的定义内，任何不在method内的代码，都默认是primary constructor的代码
    class CarefulGreeter(greeting: String) {
      if (greeting == null) {
        throw new NullPointerException("greeting was null")
      }
      def greet() = println(greeting)
    }
    new CarefulGreeter(null)

### 2.8. Function

    //function declaration
    def functionName ([list of parameters]) [:return type]
    
    //function definition
     def functionName ([list of parameters]) [:return type] = {
       function body
       return [expr]
    }
    
    /*
    如果函数体内有return表达式，则[:return type]不可省略。
    如果没有return表达式，则[:return type]可省略，而把最后一个表达式的值默认为返回值
    */
    def hello(name:String="Spark"):String ={
        return "hello" +name
    }
    def hello(name:String="Spark")={
        "hello" +name
    }
    
    //函数体内只有一个表达式时，可以省略{}
    def square(x: Int): Int =
       x * x           
    
    //对于递归函数，不可以省略: return type
    //如果省略=，则函数不返回值（即返回的类型为Unit）
    
    /*匿名函数*/
    var mul = (x:Int, y:Int) => x*y
    mul(2,3)
    
### 2.9. Array
Array是可变对象，元素需要未相同类型。

    var z = new Array[String](3)
    var z = Array("1","123","asd")
     
    z(0) //访问第一个元素，值为"1"
    z(1) //值为"123"
     
     
    z(0) = "heh"
     
     
    z.distinct
    z.filter(x => x>0) //过滤掉Array中小于等于0的元素（使用匿名函数）
    z.filter(x => x<0) //过滤掉Array中大于等于0的元素
    
### 2.10. Collection

Collectin分为两种：

- List：不可变对象。元素必须是相同类型
- Tuple：不可变对象。元素无须是相同类型

举例：

    // Define List of integers.
    val x = List(1,2,3,4)
    x(0) //访问List的第一个元素，值为1
     
    // Define a set.
    var x = Set(1,3,5,7)
     
    // Define a map.
    val x = Map("one" -> 1, "two" -> 2, "three" -> 3)
     
    // Create a tuple of two elements.
    val x = (10, "Scala")
    x._1 //1-based,访问Tuple的第一个元素，值为10
    x._2 //访问Tuple的第二个元素，值为"Scala"




