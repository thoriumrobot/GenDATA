/*
    @Positive
 * Copyright (c) 2004, 2019, Oracle and/or its affiliates. All rights reserved.
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
package sun.management;

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
import java.io.Serializable;
    @Positive
import java.util.*;
    @Positive
import javax.management.openmbean.ArrayType;
    @Positive
import javax.management.openmbean.CompositeData;
    @Positive
import javax.management.openmbean.CompositeType;
    @Positive
import javax.management.openmbean.OpenType;
    @Positive
import javax.management.openmbean.TabularType;

    @Positive
public abstract class LazyCompositeData implements CompositeData, Serializable {

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    public boolean containsKey(String key);

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    public boolean containsValue(Object value);

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    @Override
    @Positive
    public Object get(String key);

    @Positive
    @Override
    @Positive
    public Object[] getAll(String[] keys);

    @Positive
    @Override
    @Positive
    public CompositeType getCompositeType();

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    @Override
    @Positive
    public Collection<?> values();

    @Positive
    protected Object writeReplace() throws java.io.ObjectStreamException;

    @Positive
    protected abstract CompositeData getCompositeData();

    @Positive
    public static String getString(CompositeData cd, String itemName);

    @Positive
    public static boolean getBoolean(CompositeData cd, String itemName);

    @Positive
    public static long getLong(CompositeData cd, String itemName);

    @Positive
    public static int getInt(CompositeData cd, String itemName);

    @Positive
    protected static boolean isTypeMatched(CompositeType type1, CompositeType type2);

    @Positive
    protected static boolean isTypeMatched(TabularType type1, TabularType type2);

    @Positive
    protected static boolean isTypeMatched(ArrayType<?> type1, ArrayType<?> type2);
    @Positive
}

// CFWR semantic augmentation - variant 0
