/*
    @Positive
 * Copyright (c) 2002, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.
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
 *
    @Positive
 */
    @Positive
package sun.jvm.hotspot.debugger;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.util.*;

    @Positive
public class LongHashMap {

    @Positive
    static class Entry {

    @Positive
        long getKey();

    @Positive
        Object getValue();

    @Positive
        Object setValue(Object value);

    @Positive
        public boolean equals(Object o);

    @Positive
        public int hashCode();
    @Positive
    }

    @Positive
    public LongHashMap(int initialCapacity, float loadFactor) {
    @Positive
    }

    @Positive
    public LongHashMap(int initialCapacity) {
    @Positive
    }

    @Positive
    public LongHashMap() {
    @Positive
    }

    @Positive
    public int size();

    @Positive
    public boolean isEmpty();

    @Positive
    public Object get(long key);

    @Positive
    @Pure
    @Positive
    public boolean containsKey(long key);

    @Positive
    Entry getEntry(long key);

    @Positive
    @Pure
    @Positive
    public boolean containsValue(Object value);

    @Positive
    public Object put(long key, Object value);

    @Positive
    public Object remove(long key);

    @Positive
    Entry removeEntryForKey(long key);

    @Positive
    void removeEntry(Entry doomed);

    @Positive
    public void clear();

    @Positive
    void rehash();

    @Positive
    static boolean eq(Object o1, Object o2);

    @Positive
    Entry newEntry(int hash, long key, Object value, Entry next);

    @Positive
    int capacity();

    @Positive
    float loadFactor();
    @Positive
}

// CFWR semantic augmentation - variant 0
