/*
 * Copyright (c) 2002, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
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
 *
 */
package sun.jvm.hotspot.debugger;

import org.checkerframework.dataflow.qual.Pure;
import java.util.*;

public class LongHashMap {

    static class Entry {

        long getKey();

        Object getValue();

        Object setValue(Object value);

        public boolean equals(Object o);

        public int hashCode();
    }

    public LongHashMap(int initialCapacity, float loadFactor) {
    }

    public LongHashMap(int initialCapacity) {
    }

    public LongHashMap() {
    }

    public int size();

    public boolean isEmpty();

    public Object get(long key);

    @Pure
    public boolean containsKey(long key);

    Entry getEntry(long key);

    @Pure
    public boolean containsValue(Object value);

    public Object put(long key, Object value);

    public Object remove(long key);

    Entry removeEntryForKey(long key);

    void removeEntry(Entry doomed);

    public void clear();

    void rehash();

    static boolean eq(Object o1, Object o2);

    Entry newEntry(int hash, long key, Object value, Entry next);

    int capacity();

    float loadFactor();
}
