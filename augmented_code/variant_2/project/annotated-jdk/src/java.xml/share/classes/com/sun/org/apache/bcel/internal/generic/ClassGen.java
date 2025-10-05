/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2017, 2021, Oracle and/or its affiliates. All rights reserved.
    @Positive
 */
    @Positive
package com.sun.org.apache.bcel.internal.generic;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.List;
    @Positive
import java.util.Objects;
    @Positive
import com.sun.org.apache.bcel.internal.Const;
    @Positive
import com.sun.org.apache.bcel.internal.classfile.AccessFlags;
    @Positive
import com.sun.org.apache.bcel.internal.classfile.AnnotationEntry;
    @Positive
import com.sun.org.apache.bcel.internal.classfile.Annotations;
    @Positive
import com.sun.org.apache.bcel.internal.classfile.Attribute;
    @Positive
import com.sun.org.apache.bcel.internal.classfile.ConstantPool;
    @Positive
import com.sun.org.apache.bcel.internal.classfile.Field;
    @Positive
import com.sun.org.apache.bcel.internal.classfile.JavaClass;
    @Positive
import com.sun.org.apache.bcel.internal.classfile.Method;
    @Positive
import com.sun.org.apache.bcel.internal.classfile.RuntimeInvisibleAnnotations;
    @Positive
import com.sun.org.apache.bcel.internal.classfile.RuntimeVisibleAnnotations;
    @Positive
import com.sun.org.apache.bcel.internal.classfile.SourceFile;
    @Positive
import com.sun.org.apache.bcel.internal.util.BCELComparator;

    @Positive
public class ClassGen extends AccessFlags implements Cloneable {

    @Positive
    public ClassGen(final String className, final String superClassName, final String fileName, final int accessFlags, final String[] interfaces, final ConstantPoolGen cp) {
    @Positive
    }

    @Positive
    public ClassGen(final String className, final String superClassName, final String fileName, final int accessFlags, final String[] interfaces) {
    @Positive
    }

    @Positive
    public ClassGen(final JavaClass clazz) {
    @Positive
    }

    @Positive
    public JavaClass getJavaClass();

    @Positive
    public void addInterface(final String name);

    @Positive
    public void removeInterface(final String name);

    @Positive
    public int getMajor();

    @Positive
    public void setMajor(final int major);

    @Positive
    public void setMinor(final int minor);

    @Positive
    public int getMinor();

    @Positive
    public void addAttribute(final Attribute a);

    @Positive
    public void addAnnotationEntry(final AnnotationEntryGen a);

    @Positive
    public void addMethod(final Method m);

    @Positive
    public void addEmptyConstructor(final int access_flags);

    @Positive
    public void addField(final Field f);

    @Positive
    @Pure
    @Positive
    public boolean containsField(final Field f);

    @Positive
    public Field containsField(final String name);

    @Positive
    public Method containsMethod(final String name, final String signature);

    @Positive
    public void removeAttribute(final Attribute a);

    @Positive
    public void removeMethod(final Method m);

    @Positive
    public void replaceMethod(final Method old, final Method new_);

    @Positive
    public void replaceField(final Field old, final Field new_);

    @Positive
    public void removeField(final Field f);

    @Positive
    public String getClassName();

    @Positive
    public String getSuperclassName();

    @Positive
    public String getFileName();

    @Positive
    public void setClassName(final String name);

    @Positive
    public void setSuperclassName(final String name);

    @Positive
    public Method[] getMethods();

    @Positive
    public void setMethods(final Method[] methods);

    @Positive
    public void setMethodAt(final Method method, final int pos);

    @Positive
    public Method getMethodAt(final int pos);

    @Positive
    public String[] getInterfaceNames();

    @Positive
    public int[] getInterfaces();

    @Positive
    public Field[] getFields();

    @Positive
    public Attribute[] getAttributes();

    @Positive
    public AnnotationEntryGen[] getAnnotationEntries();

    @Positive
    public ConstantPoolGen getConstantPool();

    @Positive
    public void setConstantPool(final ConstantPoolGen constant_pool);

    @Positive
    public void setClassNameIndex(final int class_name_index);

    @Positive
    public void setSuperclassNameIndex(final int superclass_name_index);

    @Positive
    public int getSuperclassNameIndex();

    @Positive
    public int getClassNameIndex();

    @Positive
    public void addObserver(final ClassObserver o);

    @Positive
    public void removeObserver(final ClassObserver o);

    @Positive
    public void update();

    @Positive
    @Override
    @Positive
    public Object clone();

    @Positive
    public static BCELComparator getComparator();

    @Positive
    public static void setComparator(final BCELComparator comparator);

    @Positive
    @Override
    @Positive
    public boolean equals(final Object obj);

    @Positive
    @Override
    @Positive
    public int hashCode();
    @Positive
}
