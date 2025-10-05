/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1999, 2020, Oracle and/or its affiliates. All rights reserved.
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
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
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
package javax.naming.directory;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.util.Vector;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import java.lang.reflect.Array;
    @Positive
import javax.naming.NamingException;
    @Positive
import javax.naming.NamingEnumeration;
    @Positive
import javax.naming.OperationNotSupportedException;

    @Positive
public class BasicAttribute implements Attribute {

    @Positive
    protected String attrID;

    @Positive
    protected transient Vector<Object> values;

    @Positive
    protected boolean ordered;

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public Object clone();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();

    @Positive
    public String toString();

    @Positive
    public BasicAttribute(String id) {
    @Positive
    }

    @Positive
    public BasicAttribute(String id, Object value) {
    @Positive
    }

    @Positive
    public BasicAttribute(String id, boolean ordered) {
    @Positive
    }

    @Positive
    public BasicAttribute(String id, Object value, boolean ordered) {
    @Positive
    }

    @Positive
    public NamingEnumeration<?> getAll() throws NamingException;

    @Positive
    public Object get() throws NamingException;

    @Positive
    public int size();

    @Positive
    public String getID();

    @Positive
    @Pure
    @Positive
    public boolean contains(Object attrVal);

    @Positive
    public boolean add(Object attrVal);

    @Positive
    public boolean remove(Object attrval);

    @Positive
    public void clear();

    @Positive
    public boolean isOrdered();

    @Positive
    public Object get(int ix) throws NamingException;

    @Positive
    public Object remove(int ix);

    @Positive
    public void add(int ix, Object attrVal);

    @Positive
    public Object set(int ix, Object attrVal);

    @Positive
    public DirContext getAttributeSyntaxDefinition() throws NamingException;

    @Positive
    public DirContext getAttributeDefinition() throws NamingException;

    @Positive
    class ValuesEnumImpl implements NamingEnumeration<Object> {

    @Positive
        public boolean hasMoreElements();

    @Positive
        public Object nextElement();

    @Positive
        public Object next() throws NamingException;

    @Positive
        public boolean hasMore() throws NamingException;

    @Positive
        public void close() throws NamingException;
    @Positive
    }
    @Positive
}
