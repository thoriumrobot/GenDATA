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
import java.util.Vector;
import java.util.Enumeration;
import java.util.NoSuchElementException;
import java.lang.reflect.Array;
import javax.naming.NamingException;
import javax.naming.NamingEnumeration;
import javax.naming.OperationNotSupportedException;

public class BasicAttribute implements Attribute {

    protected String attrID;

    protected transient Vector<Object> values;

    protected boolean ordered;

    @SuppressWarnings("unchecked")
    public Object clone();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int hashCode();

    public String toString();

    public BasicAttribute(String id) {
    }

    public BasicAttribute(String id, Object value) {
    }

    public BasicAttribute(String id, boolean ordered) {
    }

    public BasicAttribute(String id, Object value, boolean ordered) {
    }

    public NamingEnumeration<?> getAll() throws NamingException;

    public Object get() throws NamingException;

    public int size();

    public String getID();

    @Pure
    public boolean contains(Object attrVal);

    public boolean add(Object attrVal);

    public boolean remove(Object attrval);

    public void clear();

    public boolean isOrdered();

    public Object get(int ix) throws NamingException;

    public Object remove(int ix);

    public void add(int ix, Object attrVal);

    public Object set(int ix, Object attrVal);

    public DirContext getAttributeSyntaxDefinition() throws NamingException;

    public DirContext getAttributeDefinition() throws NamingException;

    class ValuesEnumImpl implements NamingEnumeration<Object> {

        public boolean hasMoreElements();

        public Object nextElement();

        public Object next() throws NamingException;

        public boolean hasMore() throws NamingException;

        public void close() throws NamingException;
    }
}
