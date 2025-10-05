/*
    @Positive
 * Copyright (c) 1999, 2018, Oracle and/or its affiliates. All rights reserved.
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
package javax.naming.directory;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.util.Vector;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import javax.naming.NamingException;
    @Positive
import javax.naming.NamingEnumeration;
    @Positive
import javax.naming.OperationNotSupportedException;

    @Positive
public interface Attribute extends Cloneable, java.io.Serializable {

    @Positive
    NamingEnumeration<?> getAll() throws NamingException;

    @Positive
    Object get() throws NamingException;

    @Positive
    int size();

    @Positive
    String getID();

    @Positive
    @Pure
    @Positive
    boolean contains(Object attrVal);

    @Positive
    boolean add(Object attrVal);

    @Positive
    boolean remove(Object attrval);

    @Positive
    void clear();

    @Positive
    DirContext getAttributeSyntaxDefinition() throws NamingException;

    @Positive
    DirContext getAttributeDefinition() throws NamingException;

    @Positive
    Object clone();

    @Positive
    boolean isOrdered();

    @Positive
    Object get(int ix) throws NamingException;

    @Positive
    Object remove(int ix);

    @Positive
    void add(int ix, Object attrVal);

    @Positive
    Object set(int ix, Object attrVal);

    @Positive
    @Deprecated
    @Positive
    @SuppressWarnings("serial")
    @Positive
    static final long serialVersionUID;
    @Positive
}

// CFWR semantic augmentation - variant 0
