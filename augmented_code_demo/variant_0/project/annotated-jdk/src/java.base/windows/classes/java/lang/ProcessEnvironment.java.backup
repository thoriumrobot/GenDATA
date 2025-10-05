/*
 * Copyright (c) 2003, 2011, Oracle and/or its affiliates. All rights reserved.
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
package java.lang;

import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import java.io.*;
import java.util.*;

final class ProcessEnvironment extends HashMap<String, String> {

    public String put(String key, String value);

    public String get(Object key);

    @Pure
    public boolean containsKey(Object key);

    @Pure
    public boolean containsValue(Object value);

    public String remove(Object key);

    private static class CheckedEntry implements Map.Entry<String, String> {

        public CheckedEntry(Map.Entry<String, String> e) {
        }

        public String getKey();

        public String getValue();

        public String setValue(String value);

        public String toString();

        public boolean equals(Object o);

        public int hashCode();
    }

    private static class CheckedEntrySet extends AbstractSet<Map.Entry<String, String>> {

        public CheckedEntrySet(Set<Map.Entry<String, String>> s) {
        }

        public int size();

        public boolean isEmpty();

        public void clear();

        public Iterator<Map.Entry<String, String>> iterator();

        @Pure
        public boolean contains(Object o);

        public boolean remove(Object o);
    }

    private static class CheckedValues extends AbstractCollection<String> {

        public CheckedValues(Collection<String> c) {
        }

        public int size();

        public boolean isEmpty();

        public void clear();

        public Iterator<String> iterator();

        @Pure
        public boolean contains(Object o);

        public boolean remove(Object o);
    }

    private static class CheckedKeySet extends AbstractSet<String> {

        public CheckedKeySet(Set<String> s) {
        }

        public int size();

        public boolean isEmpty();

        public void clear();

        public Iterator<String> iterator();

        @Pure
        public boolean contains(Object o);

        public boolean remove(Object o);
    }

    public Set<String> keySet();

    public Collection<String> values();

    public Set<Map.Entry<String, String>> entrySet();

    private static final class NameComparator implements Comparator<String> {

        public int compare(String s1, String s2);
    }

    private static final class EntryComparator implements Comparator<Map.Entry<String, String>> {

        public int compare(Map.Entry<String, String> e1, Map.Entry<String, String> e2);
    }

    static String getenv(String name);

    static Map<String, String> getenv();

    @SuppressWarnings("unchecked")
    static Map<String, String> environment();

    static Map<String, String> emptyEnvironment(int capacity);

    String toEnvironmentBlock();

    static String toEnvironmentBlock(Map<String, String> map);
}
