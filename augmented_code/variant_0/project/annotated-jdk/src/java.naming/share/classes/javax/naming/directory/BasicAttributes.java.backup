/*
 * Copyright (c) 1999, 2020, Oracle and/or its affiliates. All rights reserved.
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
package javax.naming.directory;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.util.Hashtable;
import java.util.Enumeration;
import java.util.Locale;
import javax.naming.NamingException;
import javax.naming.NamingEnumeration;

public class BasicAttributes implements Attributes {

    public BasicAttributes() {
    }

    public BasicAttributes(boolean ignoreCase) {
    }

    public BasicAttributes(String attrID, Object val) {
    }

    public BasicAttributes(String attrID, Object val, boolean ignoreCase) {
    }

    @SuppressWarnings("unchecked")
    public Object clone();

    public boolean isCaseIgnored();

    public int size();

    public Attribute get(String attrID);

    public NamingEnumeration<Attribute> getAll();

    public NamingEnumeration<String> getIDs();

    public Attribute put(String attrID, Object val);

    public Attribute put(Attribute attr);

    public Attribute remove(String attrID);

    public String toString();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int hashCode();

    class AttrEnumImpl implements NamingEnumeration<Attribute> {

        public AttrEnumImpl() {
        }

        public boolean hasMoreElements();

        public Attribute nextElement();

        public boolean hasMore() throws NamingException;

        public Attribute next() throws NamingException;

        public void close() throws NamingException;
    }

    class IDEnumImpl implements NamingEnumeration<String> {

        public IDEnumImpl() {
        }

        public boolean hasMoreElements();

        public String nextElement();

        public boolean hasMore() throws NamingException;

        public String next() throws NamingException;

        public void close() throws NamingException;
    }
}
