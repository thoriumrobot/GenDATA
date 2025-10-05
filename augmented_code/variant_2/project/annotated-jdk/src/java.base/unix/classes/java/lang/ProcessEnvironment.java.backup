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

final class ProcessEnvironment {

    static String getenv(String name);

    static Map<String, String> getenv();

    @SuppressWarnings("unchecked")
    static Map<String, String> environment();

    static Map<String, String> emptyEnvironment(int capacity);

    private abstract static class ExternalData {

        protected final String str;

        protected final byte[] bytes;

        protected ExternalData(String str, byte[] bytes) {
        }

        public byte[] getBytes();

        public String toString();

        public boolean equals(Object o);

        public int hashCode();
    }

    private static class Variable extends ExternalData implements Comparable<Variable> {

        protected Variable(String str, byte[] bytes) {
        }

        public static Variable valueOfQueryOnly(Object str);

        public static Variable valueOfQueryOnly(String str);

        public static Variable valueOf(String str);

        public static Variable valueOf(byte[] bytes);

        public int compareTo(Variable variable);

        public boolean equals(Object o);
    }

    private static class Value extends ExternalData implements Comparable<Value> {

        protected Value(String str, byte[] bytes) {
        }

        public static Value valueOfQueryOnly(Object str);

        public static Value valueOfQueryOnly(String str);

        public static Value valueOf(String str);

        public static Value valueOf(byte[] bytes);

        public int compareTo(Value value);

        public boolean equals(Object o);
    }

    private static class StringEnvironment extends AbstractMap<String, String> {

        public StringEnvironment(Map<Variable, Value> m) {
        }

        @Pure
        public int size();

        @Pure
        public boolean isEmpty();

        public void clear();

        @Pure
        public boolean containsKey(Object key);

        @Pure
        public boolean containsValue(Object value);

        public String get(Object key);

        public String put(String key, String value);

        public String remove(Object key);

        public Set<String> keySet();

        public Set<Map.Entry<String, String>> entrySet();

        public Collection<String> values();

        public byte[] toEnvironmentBlock(int[] envc);
    }

    static byte[] toEnvironmentBlock(Map<String, String> map, int[] envc);

    private static class StringEntry implements Map.Entry<String, String> {

        public StringEntry(Map.Entry<Variable, Value> e) {
        }

        public String getKey();

        public String getValue();

        public String setValue(String newValue);

        public String toString();

        public boolean equals(Object o);

        public int hashCode();
    }

    private static class StringEntrySet extends AbstractSet<Map.Entry<String, String>> {

        public StringEntrySet(Set<Map.Entry<Variable, Value>> s) {
        }

        public int size();

        public boolean isEmpty();

        public void clear();

        public Iterator<Map.Entry<String, String>> iterator();

        @Pure
        public boolean contains(Object o);

        public boolean remove(Object o);

        @Pure
        public boolean equals(Object o);

        @Pure
        public int hashCode();
    }

    private static class StringValues extends AbstractCollection<String> {

        public StringValues(Collection<Value> c) {
        }

        public int size();

        public boolean isEmpty();

        public void clear();

        public Iterator<String> iterator();

        @Pure
        public boolean contains(Object o);

        public boolean remove(Object o);

        public boolean equals(Object o);

        @Pure
        public int hashCode();
    }

    private static class StringKeySet extends AbstractSet<String> {

        public StringKeySet(Set<Variable> s) {
        }

        public int size();

        public boolean isEmpty();

        public void clear();

        public Iterator<String> iterator();

        @Pure
        public boolean contains(Object o);

        public boolean remove(Object o);
    }
}
