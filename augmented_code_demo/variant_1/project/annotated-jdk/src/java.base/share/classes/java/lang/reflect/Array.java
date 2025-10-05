/*
    @Positive
 * Copyright (c) 1996, 2020, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package java.lang.reflect;

    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.LengthOf;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.common.value.qual.StaticallyExecutable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import jdk.internal.vm.annotation.IntrinsicCandidate;

    @Positive
@AnnotatedFor({ "index", "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public final class Array {

    @Positive
    @SideEffectFree
    @Positive
    public static Object newInstance(Class<?> componentType, @NonNegative int length) throws NegativeArraySizeException;

    @Positive
    @SideEffectFree
    @Positive
    public static Object newInstance(Class<?> componentType, @NonNegative int... dimensions) throws IllegalArgumentException, NegativeArraySizeException;

    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    @StaticallyExecutable
    @Positive
    @LengthOf({ "#1" })
    @Positive
    public static native int getLength(Object array) throws IllegalArgumentException;

    @Positive
    @Pure
    @Positive
    public static native Object get(Object array, @IndexFor({ "#1" }) int index) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static native boolean getBoolean(Object array, @IndexFor({ "#1" }) int index) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static native byte getByte(Object array, @IndexFor({ "#1" }) int index) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static native char getChar(Object array, @IndexFor({ "#1" }) int index) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static native short getShort(Object array, @IndexFor({ "#1" }) int index) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static native int getInt(Object array, @IndexFor({ "#1" }) int index) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static native long getLong(Object array, @IndexFor({ "#1" }) int index) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static native float getFloat(Object array, @IndexFor({ "#1" }) int index) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static native double getDouble(Object array, @IndexFor({ "#1" }) int index) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    @Positive
    public static native void set(Object array, @IndexFor({ "#1" }) int index, Object value) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    @Positive
    public static native void setBoolean(Object array, @IndexFor({ "#1" }) int index, boolean z) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    @Positive
    public static native void setByte(Object array, @IndexFor({ "#1" }) int index, byte b) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    @Positive
    public static native void setChar(Object array, @IndexFor({ "#1" }) int index, char c) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    @Positive
    public static native void setShort(Object array, @IndexFor({ "#1" }) int index, short s) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    @Positive
    public static native void setInt(Object array, @IndexFor({ "#1" }) int index, int i) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    @Positive
    public static native void setLong(Object array, @IndexFor({ "#1" }) int index, long l) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    @Positive
    public static native void setFloat(Object array, @IndexFor({ "#1" }) int index, float f) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    @Positive
    public static native void setDouble(Object array, @IndexFor({ "#1" }) int index, double d) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;
    @Positive
}

// CFWR semantic augmentation - variant 1
