/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
final class ProcessEnvironment {

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
    private abstract static class ExternalData {

    @Positive
        protected final String str;

    @Positive
        protected final byte[] bytes;

    @Positive
        protected ExternalData(String str, byte[] bytes) {
    @Positive
        }

    @Positive
        public byte[] getBytes();

    @Positive
        public String toString();

    @Positive
        public boolean equals(Object o);

    @Positive
        public int hashCode();
    @Positive
    }

    @Positive
    private static class Variable extends ExternalData implements Comparable<Variable> {

    @Positive
        protected Variable(String str, byte[] bytes) {
    @Positive
        }

    @Positive
        public static Variable valueOfQueryOnly(Object str);

    @Positive
        public static Variable valueOfQueryOnly(String str);

    @Positive
        public static Variable valueOf(String str);

    @Positive
        public static Variable valueOf(byte[] bytes);

    @Positive
        public int compareTo(Variable variable);

    @Positive
        public boolean equals(Object o);
    @Positive
    }

    @Positive
    private static class Value extends ExternalData implements Comparable<Value> {

    @Positive
        protected Value(String str, byte[] bytes) {
    @Positive
        }

    @Positive
        public static Value valueOfQueryOnly(Object str);

    @Positive
        public static Value valueOfQueryOnly(String str);

    @Positive
        public static Value valueOf(String str);

    @Positive
        public static Value valueOf(byte[] bytes);

    @Positive
        public int compareTo(Value value);

    @Positive
        public boolean equals(Object o);
    @Positive
    }

    @Positive
    private static class StringEnvironment extends AbstractMap<String, String> {

    @Positive
        public StringEnvironment(Map<Variable, Value> m) {
    @Positive
        }

    @Positive
        @Pure
    @Positive
        public int size();

    @Positive
        @Pure
    @Positive
        public boolean isEmpty();

    @Positive
        public void clear();

    @Positive
        @Pure
    @Positive
        public boolean containsKey(Object key);

    @Positive
        @Pure
    @Positive
        public boolean containsValue(Object value);

    @Positive
        public String get(Object key);

    @Positive
        public String put(String key, String value);

    @Positive
        public String remove(Object key);

    @Positive
        public Set<String> keySet();

    @Positive
        public Set<Map.Entry<String, String>> entrySet();

    @Positive
        public Collection<String> values();

    @Positive
        public byte[] toEnvironmentBlock(int[] envc);
    @Positive
    }

    @Positive
    static byte[] toEnvironmentBlock(Map<String, String> map, int[] envc);

    @Positive
    private static class StringEntry implements Map.Entry<String, String> {

    @Positive
        public StringEntry(Map.Entry<Variable, Value> e) {
    @Positive
        }

    @Positive
        public String getKey();

    @Positive
        public String getValue();

    @Positive
        public String setValue(String newValue);

    @Positive
        public String toString();

    @Positive
        public boolean equals(Object o);

    @Positive
        public int hashCode();
    @Positive
    }

    @Positive
    private static class StringEntrySet extends AbstractSet<Map.Entry<String, String>> {

    @Positive
        public StringEntrySet(Set<Map.Entry<Variable, Value>> s) {
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
        @Pure
    @Positive
        public boolean equals(Object o);

    @Positive
        @Pure
    @Positive
        public int hashCode();
    @Positive
    }

    @Positive
    private static class StringValues extends AbstractCollection<String> {

    @Positive
        public StringValues(Collection<Value> c) {
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
        public boolean equals(Object o);

    @Positive
        @Pure
    @Positive
        public int hashCode();
    @Positive
    }

    @Positive
    private static class StringKeySet extends AbstractSet<String> {

    @Positive
        public StringKeySet(Set<Variable> s) {
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
}
