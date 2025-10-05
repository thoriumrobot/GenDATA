/*
 * Copyright (c) 2005, 2013, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.net.httpserver;

import org.checkerframework.dataflow.qual.Pure;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Headers implements Map<String, List<String>> {

    public Headers() {
    }

    public int size();

    public boolean isEmpty();

    @Pure
    public boolean containsKey(Object key);

    @Pure
    public boolean containsValue(Object value);

    public List<String> get(Object key);

    public String getFirst(String key);

    public List<String> put(String key, List<String> value);

    public void add(String key, String value);

    public void set(String key, String value);

    public List<String> remove(Object key);

    public void putAll(Map<? extends String, ? extends List<String>> t);

    public void clear();

    public Set<String> keySet();

    public Collection<List<String>> values();

    public Set<Map.Entry<String, List<String>>> entrySet();

    public boolean equals(Object o);

    public int hashCode();
}
