/*
 * Copyright (c) 2000, 2011, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.jndi.dns;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Enumeration;
import javax.naming.*;

public final class DnsName implements Name {

    public DnsName() {
    }

    public DnsName(String name) throws InvalidNameException {
    }

    public String toString();

    public boolean isHostName();

    public short getOctets();

    public int size();

    public boolean isEmpty();

    public int hashCode();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int compareTo(Object obj);

    public boolean startsWith(Name n);

    public boolean endsWith(Name n);

    public String get(int pos);

    public Enumeration<String> getAll();

    public Name getPrefix(int pos);

    public Name getSuffix(int pos);

    public Object clone();

    public Object remove(int pos);

    public Name add(String comp) throws InvalidNameException;

    public Name add(int pos, String comp) throws InvalidNameException;

    public Name addAll(Name suffix) throws InvalidNameException;

    public Name addAll(int pos, Name n) throws InvalidNameException;

    boolean hasRootLabel();

    String getKey(int i);
}
