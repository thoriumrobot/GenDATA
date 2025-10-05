/*
 * Copyright (c) 1999, 2018, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.dataflow.qual.Pure;
import java.util.Vector;
import java.util.Enumeration;
import java.util.NoSuchElementException;
import javax.naming.NamingException;
import javax.naming.NamingEnumeration;
import javax.naming.OperationNotSupportedException;

public interface Attribute extends Cloneable, java.io.Serializable {

    NamingEnumeration<?> getAll() throws NamingException;

    Object get() throws NamingException;

    int size();

    String getID();

    @Pure
    boolean contains(Object attrVal);

    boolean add(Object attrVal);

    boolean remove(Object attrval);

    void clear();

    DirContext getAttributeSyntaxDefinition() throws NamingException;

    DirContext getAttributeDefinition() throws NamingException;

    Object clone();

    boolean isOrdered();

    Object get(int ix) throws NamingException;

    Object remove(int ix);

    void add(int ix, Object attrVal);

    Object set(int ix, Object attrVal);

    @Deprecated
    @SuppressWarnings("serial")
    static final long serialVersionUID;
}
