/*
    @Positive
 * Copyright (c) 2000, 2011, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.jndi.dns;

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
import java.util.ArrayList;
    @Positive
import java.util.Comparator;
    @Positive
import java.util.Enumeration;
    @Positive
import javax.naming.*;

    @Positive
public final class DnsName implements Name {

    @Positive
    public DnsName() {
    @Positive
    }

    @Positive
    public DnsName(String name) throws InvalidNameException {
    @Positive
    }

    @Positive
    public String toString();

    @Positive
    public boolean isHostName();

    @Positive
    public short getOctets();

    @Positive
    public int size();

    @Positive
    public boolean isEmpty();

    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int compareTo(Object obj);

    @Positive
    public boolean startsWith(Name n);

    @Positive
    public boolean endsWith(Name n);

    @Positive
    public String get(int pos);

    @Positive
    public Enumeration<String> getAll();

    @Positive
    public Name getPrefix(int pos);

    @Positive
    public Name getSuffix(int pos);

    @Positive
    public Object clone();

    @Positive
    public Object remove(int pos);

    @Positive
    public Name add(String comp) throws InvalidNameException;

    @Positive
    public Name add(int pos, String comp) throws InvalidNameException;

    @Positive
    public Name addAll(Name suffix) throws InvalidNameException;

    @Positive
    public Name addAll(int pos, Name n) throws InvalidNameException;

    @Positive
    boolean hasRootLabel();

    @Positive
    String getKey(int i);
    @Positive
}

// CFWR semantic augmentation - variant 1
