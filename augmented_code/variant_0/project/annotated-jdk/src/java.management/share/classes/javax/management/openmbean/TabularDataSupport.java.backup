/*
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
import com.sun.jmx.mbeanserver.GetPropertyAction;
import com.sun.jmx.mbeanserver.Util;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.security.AccessController;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import jdk.internal.access.SharedSecrets;

public class TabularDataSupport implements TabularData, Map<Object, Object>, Cloneable, Serializable {

    public TabularDataSupport(TabularType tabularType) {
    }

    public TabularDataSupport(TabularType tabularType, int initialCapacity, float loadFactor) {
    }

    public TabularType getTabularType();

    public Object[] calculateIndex(CompositeData value);

    @Pure
    public boolean containsKey(Object key);

    @Pure
    public boolean containsKey(Object[] key);

    @Pure
    public boolean containsValue(CompositeData value);

    @Pure
    public boolean containsValue(Object value);

    public Object get(Object key);

    public CompositeData get(Object[] key);

    public Object put(Object key, Object value);

    public void put(CompositeData value);

    public Object remove(Object key);

    public CompositeData remove(Object[] key);

    public void putAll(Map<?, ?> t);

    public void putAll(CompositeData[] values);

    public void clear();

    public int size();

    public boolean isEmpty();

    public Set<Object> keySet();

    @SuppressWarnings("unchecked")
    public Collection<Object> values();

    @SuppressWarnings("unchecked")
    public Set<Map.Entry<Object, Object>> entrySet();

    public Object clone();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int hashCode();

    public String toString();
}
