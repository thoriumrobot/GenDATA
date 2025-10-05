/*
 * Copyright (c) 1999, 2011, Oracle and/or its affiliates. All rights reserved.
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
import java.util.Locale;
import java.util.Vector;
import java.util.Enumeration;
import java.util.Properties;
import java.util.NoSuchElementException;

class NameImpl {

    public String toString();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int compareTo(NameImpl obj);

    public int size();

    public Enumeration<String> getAll();

    public String get(int posn);

    public Enumeration<String> getPrefix(int posn);

    public Enumeration<String> getSuffix(int posn);

    public boolean isEmpty();

    public boolean startsWith(int posn, Enumeration<String> prefix);

    public boolean endsWith(int posn, Enumeration<String> suffix);

    public boolean addAll(Enumeration<String> comps) throws InvalidNameException;

    public boolean addAll(int posn, Enumeration<String> comps) throws InvalidNameException;

    public void add(String comp) throws InvalidNameException;

    public void add(int posn, String comp) throws InvalidNameException;

    public Object remove(int posn);

    public int hashCode();
}

final class NameImplEnumerator implements Enumeration<String> {

    public boolean hasMoreElements();

    public String nextElement();
}
