/*
    @Positive
 * Copyright (c) 2000, 2015, Oracle and/or its affiliates. All rights reserved.
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
package javax.management.openmbean;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.io.ObjectStreamException;
    @Positive
import java.lang.reflect.Array;

    @Positive
public class ArrayType<T> extends OpenType<T> {

    @Positive
    static boolean isPrimitiveContentType(final String primitiveKey);

    @Positive
    static String getPrimitiveTypeKey(String elementClassName);

    @Positive
    static String getPrimitiveTypeName(String elementClassName);

    @Positive
    static SimpleType<?> getPrimitiveOpenType(String primitiveTypeName);

    @Positive
    public ArrayType(int dimension, OpenType<?> elementType) throws OpenDataException {
    @Positive
    }

    @Positive
    public ArrayType(SimpleType<?> elementType, boolean primitiveArray) throws OpenDataException {
    @Positive
    }

    @Positive
    public int getDimension();

    @Positive
    public OpenType<?> getElementOpenType();

    @Positive
    public boolean isPrimitiveArray();

    @Positive
    public boolean isValue(Object obj);

    @Positive
    @Override
    @Positive
    boolean isAssignableFrom(OpenType<?> ot);

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();

    @Positive
    public String toString();

    @Positive
    public static <E> ArrayType<E[]> getArrayType(OpenType<E> elementType) throws OpenDataException;

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public static <T> ArrayType<T> getPrimitiveArrayType(Class<T> arrayClass);
    @Positive
}

// CFWR semantic augmentation - variant 0
