/*
 * Copyright (c) 1996, 2020, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package java.lang.reflect;

import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.LengthOf;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.common.value.qual.StaticallyExecutable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import jdk.internal.vm.annotation.IntrinsicCandidate;

@AnnotatedFor({ "index", "interning" })
@UsesObjectEquals
public final class Array {

    @SideEffectFree
    public static Object newInstance(Class<?> componentType, @NonNegative int length) throws NegativeArraySizeException;

    @SideEffectFree
    public static Object newInstance(Class<?> componentType, @NonNegative int... dimensions) throws IllegalArgumentException, NegativeArraySizeException;

    @Pure
    @IntrinsicCandidate
    @StaticallyExecutable
    @LengthOf({ "#1" })
    public static native int getLength(Object array) throws IllegalArgumentException;

    @Pure
    public static native Object get(Object array, @IndexFor({ "#1" }) int index) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    @Pure
    @StaticallyExecutable
    public static native boolean getBoolean(Object array, @IndexFor({ "#1" }) int index) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    @Pure
    @StaticallyExecutable
    public static native byte getByte(Object array, @IndexFor({ "#1" }) int index) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    @Pure
    @StaticallyExecutable
    public static native char getChar(Object array, @IndexFor({ "#1" }) int index) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    @Pure
    @StaticallyExecutable
    public static native short getShort(Object array, @IndexFor({ "#1" }) int index) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    @Pure
    @StaticallyExecutable
    public static native int getInt(Object array, @IndexFor({ "#1" }) int index) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    @Pure
    @StaticallyExecutable
    public static native long getLong(Object array, @IndexFor({ "#1" }) int index) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    @Pure
    @StaticallyExecutable
    public static native float getFloat(Object array, @IndexFor({ "#1" }) int index) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    @Pure
    @StaticallyExecutable
    public static native double getDouble(Object array, @IndexFor({ "#1" }) int index) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    public static native void set(Object array, @IndexFor({ "#1" }) int index, Object value) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    public static native void setBoolean(Object array, @IndexFor({ "#1" }) int index, boolean z) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    public static native void setByte(Object array, @IndexFor({ "#1" }) int index, byte b) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    public static native void setChar(Object array, @IndexFor({ "#1" }) int index, char c) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    public static native void setShort(Object array, @IndexFor({ "#1" }) int index, short s) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    public static native void setInt(Object array, @IndexFor({ "#1" }) int index, int i) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    public static native void setLong(Object array, @IndexFor({ "#1" }) int index, long l) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    public static native void setFloat(Object array, @IndexFor({ "#1" }) int index, float f) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;

    public static native void setDouble(Object array, @IndexFor({ "#1" }) int index, double d) throws IllegalArgumentException, ArrayIndexOutOfBoundsException;
}
