/*
    @Positive
 * Copyright (c) 2003, 2011, Oracle and/or its affiliates. All rights reserved.
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
package java.lang;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import java.io.*;
    @Positive
import java.util.*;

    @Positive
final class ProcessEnvironment extends HashMap<String, String> {

    @Positive
    public String put(String key, String value);

    @Positive
    public String get(Object key);

    @Positive
    @Pure
    @Positive
    public boolean containsKey(Object key);

    @Positive
    @Pure
    @Positive
    public boolean containsValue(Object value);

    @Positive
    public String remove(Object key);

    @Positive
    private static class CheckedEntry implements Map.Entry<String, String> {

    @Positive
        public CheckedEntry(Map.Entry<String, String> e) {
    @Positive
        }

    @Positive
        public String getKey();

    @Positive
        public String getValue();

    @Positive
        public String setValue(String value);

    @Positive
        public String toString();

    @Positive
        public boolean equals(Object o);

    @Positive
        public int hashCode();
    @Positive
    }

    @Positive
    private static class CheckedEntrySet extends AbstractSet<Map.Entry<String, String>> {

    @Positive
        public CheckedEntrySet(Set<Map.Entry<String, String>> s) {
    @Positive
        }

    @Positive
        public int size();

    @Positive
        public boolean isEmpty();

    @Positive
        public void clear();

    @Positive
        public Iterator<Map.Entry<String, String>> iterator();

    @Positive
        @Pure
    @Positive
        public boolean contains(Object o);

    @Positive
        public boolean remove(Object o);
    @Positive
    }

    @Positive
    private static class CheckedValues extends AbstractCollection<String> {

    @Positive
        public CheckedValues(Collection<String> c) {
    @Positive
        }

    @Positive
        public int size();

    @Positive
        public boolean isEmpty();

    @Positive
        public void clear();

    @Positive
        public Iterator<String> iterator();

    @Positive
        @Pure
    @Positive
        public boolean contains(Object o);

    @Positive
        public boolean remove(Object o);
    @Positive
    }

    @Positive
    private static class CheckedKeySet extends AbstractSet<String> {

    @Positive
        public CheckedKeySet(Set<String> s) {
    @Positive
        }

    @Positive
        public int size();

    @Positive
        public boolean isEmpty();

    @Positive
        public void clear();

    @Positive
        public Iterator<String> iterator();

    @Positive
        @Pure
    @Positive
        public boolean contains(Object o);

    @Positive
        public boolean remove(Object o);
    @Positive
    }

    @Positive
    public Set<String> keySet();

    @Positive
    public Collection<String> values();

    @Positive
    public Set<Map.Entry<String, String>> entrySet();

    @Positive
    private static final class NameComparator implements Comparator<String> {

    @Positive
        public int compare(String s1, String s2);
    @Positive
    }

    @Positive
    private static final class EntryComparator implements Comparator<Map.Entry<String, String>> {

    @Positive
        public int compare(Map.Entry<String, String> e1, Map.Entry<String, String> e2);
    @Positive
    }

    @Positive
    static String getenv(String name);

    @Positive
    static Map<String, String> getenv();

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    static Map<String, String> environment();

    @Positive
    static Map<String, String> emptyEnvironment(int capacity);

    @Positive
    String toEnvironmentBlock();

    @Positive
    static String toEnvironmentBlock(Map<String, String> map);
    @Positive
}

// CFWR semantic augmentation - variant 1
