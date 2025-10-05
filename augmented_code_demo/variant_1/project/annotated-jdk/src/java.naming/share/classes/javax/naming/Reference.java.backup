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
package javax.naming;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.util.Vector;
import java.util.Enumeration;

public class Reference implements Cloneable, java.io.Serializable {

    protected String className;

    protected Vector<RefAddr> addrs;

    protected String classFactory;

    protected String classFactoryLocation;

    public Reference(String className) {
    }

    public Reference(String className, RefAddr addr) {
    }

    public Reference(String className, String factory, String factoryLocation) {
    }

    public Reference(String className, RefAddr addr, String factory, String factoryLocation) {
    }

    public String getClassName();

    public String getFactoryClassName();

    public String getFactoryClassLocation();

    public RefAddr get(String addrType);

    public RefAddr get(int posn);

    public Enumeration<RefAddr> getAll();

    public int size();

    public void add(RefAddr addr);

    public void add(int posn, RefAddr addr);

    public Object remove(int posn);

    public void clear();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int hashCode();

    public String toString();

    public Object clone();
}
