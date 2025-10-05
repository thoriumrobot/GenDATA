/*
 * Copyright (c) 2000, 2015, Oracle and/or its affiliates. All rights reserved.
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
package javax.management.openmbean;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.io.ObjectStreamException;
import java.lang.reflect.Array;

public class ArrayType<T> extends OpenType<T> {

    static boolean isPrimitiveContentType(final String primitiveKey);

    static String getPrimitiveTypeKey(String elementClassName);

    static String getPrimitiveTypeName(String elementClassName);

    static SimpleType<?> getPrimitiveOpenType(String primitiveTypeName);

    public ArrayType(int dimension, OpenType<?> elementType) throws OpenDataException {
    }

    public ArrayType(SimpleType<?> elementType, boolean primitiveArray) throws OpenDataException {
    }

    public int getDimension();

    public OpenType<?> getElementOpenType();

    public boolean isPrimitiveArray();

    public boolean isValue(Object obj);

    @Override
    boolean isAssignableFrom(OpenType<?> ot);

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int hashCode();

    public String toString();

    public static <E> ArrayType<E[]> getArrayType(OpenType<E> elementType) throws OpenDataException;

    @SuppressWarnings("unchecked")
    public static <T> ArrayType<T> getPrimitiveArrayType(Class<T> arrayClass);
}
