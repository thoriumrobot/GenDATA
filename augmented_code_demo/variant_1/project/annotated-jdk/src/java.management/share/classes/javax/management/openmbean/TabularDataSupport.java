/*
    @Positive
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
import com.sun.jmx.mbeanserver.GetPropertyAction;
    @Positive
import com.sun.jmx.mbeanserver.Util;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.Serializable;
    @Positive
import java.security.AccessController;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Collections;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.LinkedHashMap;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.Set;
    @Positive
import jdk.internal.access.SharedSecrets;

    @Positive
public class TabularDataSupport implements TabularData, Map<Object, Object>, Cloneable, Serializable {

    @Positive
    public TabularDataSupport(TabularType tabularType) {
    @Positive
    }

    @Positive
    public TabularDataSupport(TabularType tabularType, int initialCapacity, float loadFactor) {
    @Positive
    }

    @Positive
    public TabularType getTabularType();

    @Positive
    public Object[] calculateIndex(CompositeData value);

    @Positive
    @Pure
    @Positive
    public boolean containsKey(Object key);

    @Positive
    @Pure
    @Positive
    public boolean containsKey(Object[] key);

    @Positive
    @Pure
    @Positive
    public boolean containsValue(CompositeData value);

    @Positive
    @Pure
    @Positive
    public boolean containsValue(Object value);

    @Positive
    public Object get(Object key);

    @Positive
    public CompositeData get(Object[] key);

    @Positive
    public Object put(Object key, Object value);

    @Positive
    public void put(CompositeData value);

    @Positive
    public Object remove(Object key);

    @Positive
    public CompositeData remove(Object[] key);

    @Positive
    public void putAll(Map<?, ?> t);

    @Positive
    public void putAll(CompositeData[] values);

    @Positive
    public void clear();

    @Positive
    public int size();

    @Positive
    public boolean isEmpty();

    @Positive
    public Set<Object> keySet();

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public Collection<Object> values();

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public Set<Map.Entry<Object, Object>> entrySet();

    @Positive
    public Object clone();

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
}

// CFWR semantic augmentation - variant 1
