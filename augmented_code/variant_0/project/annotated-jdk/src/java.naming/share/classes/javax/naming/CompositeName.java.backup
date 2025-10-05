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
import java.util.Enumeration;
import java.util.Properties;

public class CompositeName implements Name {

    protected CompositeName(Enumeration<String> comps) {
    }

    public CompositeName(String n) throws InvalidNameException {
    }

    public CompositeName() {
    }

    public String toString();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int hashCode();

    public int compareTo(Object obj);

    public Object clone();

    public int size();

    public boolean isEmpty();

    public Enumeration<String> getAll();

    public String get(int posn);

    public Name getPrefix(int posn);

    public Name getSuffix(int posn);

    public boolean startsWith(Name n);

    public boolean endsWith(Name n);

    public Name addAll(Name suffix) throws InvalidNameException;

    public Name addAll(int posn, Name n) throws InvalidNameException;

    public Name add(String comp) throws InvalidNameException;

    public Name add(int posn, String comp) throws InvalidNameException;

    public Object remove(int posn) throws InvalidNameException;
}
