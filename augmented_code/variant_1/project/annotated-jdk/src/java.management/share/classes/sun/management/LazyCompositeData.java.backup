/*
 * Copyright (c) 2004, 2019, Oracle and/or its affiliates. All rights reserved.
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
package sun.management;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.io.Serializable;
import java.util.*;
import javax.management.openmbean.ArrayType;
import javax.management.openmbean.CompositeData;
import javax.management.openmbean.CompositeType;
import javax.management.openmbean.OpenType;
import javax.management.openmbean.TabularType;

public abstract class LazyCompositeData implements CompositeData, Serializable {

    @Override
    @Pure
    public boolean containsKey(String key);

    @Override
    @Pure
    public boolean containsValue(Object value);

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @Override
    public Object get(String key);

    @Override
    public Object[] getAll(String[] keys);

    @Override
    public CompositeType getCompositeType();

    @Override
    public int hashCode();

    @Override
    public String toString();

    @Override
    public Collection<?> values();

    protected Object writeReplace() throws java.io.ObjectStreamException;

    protected abstract CompositeData getCompositeData();

    public static String getString(CompositeData cd, String itemName);

    public static boolean getBoolean(CompositeData cd, String itemName);

    public static long getLong(CompositeData cd, String itemName);

    public static int getInt(CompositeData cd, String itemName);

    protected static boolean isTypeMatched(CompositeType type1, CompositeType type2);

    protected static boolean isTypeMatched(TabularType type1, TabularType type2);

    protected static boolean isTypeMatched(ArrayType<?> type1, ArrayType<?> type2);
}
