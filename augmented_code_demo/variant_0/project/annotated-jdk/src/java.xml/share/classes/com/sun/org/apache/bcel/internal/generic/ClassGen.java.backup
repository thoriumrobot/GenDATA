/*
 * Copyright (c) 2017, 2021, Oracle and/or its affiliates. All rights reserved.
 */
package com.sun.org.apache.bcel.internal.generic;

import org.checkerframework.dataflow.qual.Pure;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import com.sun.org.apache.bcel.internal.Const;
import com.sun.org.apache.bcel.internal.classfile.AccessFlags;
import com.sun.org.apache.bcel.internal.classfile.AnnotationEntry;
import com.sun.org.apache.bcel.internal.classfile.Annotations;
import com.sun.org.apache.bcel.internal.classfile.Attribute;
import com.sun.org.apache.bcel.internal.classfile.ConstantPool;
import com.sun.org.apache.bcel.internal.classfile.Field;
import com.sun.org.apache.bcel.internal.classfile.JavaClass;
import com.sun.org.apache.bcel.internal.classfile.Method;
import com.sun.org.apache.bcel.internal.classfile.RuntimeInvisibleAnnotations;
import com.sun.org.apache.bcel.internal.classfile.RuntimeVisibleAnnotations;
import com.sun.org.apache.bcel.internal.classfile.SourceFile;
import com.sun.org.apache.bcel.internal.util.BCELComparator;

public class ClassGen extends AccessFlags implements Cloneable {

    public ClassGen(final String className, final String superClassName, final String fileName, final int accessFlags, final String[] interfaces, final ConstantPoolGen cp) {
    }

    public ClassGen(final String className, final String superClassName, final String fileName, final int accessFlags, final String[] interfaces) {
    }

    public ClassGen(final JavaClass clazz) {
    }

    public JavaClass getJavaClass();

    public void addInterface(final String name);

    public void removeInterface(final String name);

    public int getMajor();

    public void setMajor(final int major);

    public void setMinor(final int minor);

    public int getMinor();

    public void addAttribute(final Attribute a);

    public void addAnnotationEntry(final AnnotationEntryGen a);

    public void addMethod(final Method m);

    public void addEmptyConstructor(final int access_flags);

    public void addField(final Field f);

    @Pure
    public boolean containsField(final Field f);

    public Field containsField(final String name);

    public Method containsMethod(final String name, final String signature);

    public void removeAttribute(final Attribute a);

    public void removeMethod(final Method m);

    public void replaceMethod(final Method old, final Method new_);

    public void replaceField(final Field old, final Field new_);

    public void removeField(final Field f);

    public String getClassName();

    public String getSuperclassName();

    public String getFileName();

    public void setClassName(final String name);

    public void setSuperclassName(final String name);

    public Method[] getMethods();

    public void setMethods(final Method[] methods);

    public void setMethodAt(final Method method, final int pos);

    public Method getMethodAt(final int pos);

    public String[] getInterfaceNames();

    public int[] getInterfaces();

    public Field[] getFields();

    public Attribute[] getAttributes();

    public AnnotationEntryGen[] getAnnotationEntries();

    public ConstantPoolGen getConstantPool();

    public void setConstantPool(final ConstantPoolGen constant_pool);

    public void setClassNameIndex(final int class_name_index);

    public void setSuperclassNameIndex(final int superclass_name_index);

    public int getSuperclassNameIndex();

    public int getClassNameIndex();

    public void addObserver(final ClassObserver o);

    public void removeObserver(final ClassObserver o);

    public void update();

    @Override
    public Object clone();

    public static BCELComparator getComparator();

    public static void setComparator(final BCELComparator comparator);

    @Override
    public boolean equals(final Object obj);

    @Override
    public int hashCode();
}
