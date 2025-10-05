/*
    @Positive
 * Copyright (c) 2003, 2020, Oracle and/or its affiliates. All rights reserved.
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
package javax.naming.ldap;

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
import javax.naming.Name;
    @Positive
import javax.naming.InvalidNameException;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.Collection;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.List;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.ListIterator;
    @Positive
import java.util.Collections;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.IOException;

    @Positive
public class LdapName implements Name {

    @Positive
    public LdapName(String name) throws InvalidNameException {
    @Positive
    }

    @Positive
    public LdapName(List<Rdn> rdns) {
    @Positive
    }

    @Positive
    public int size();

    @Positive
    public boolean isEmpty();

    @Positive
    public Enumeration<String> getAll();

    @Positive
    public String get(int posn);

    @Positive
    public Rdn getRdn(int posn);

    @Positive
    public Name getPrefix(int posn);

    @Positive
    public Name getSuffix(int posn);

    @Positive
    public boolean startsWith(Name n);

    @Positive
    public boolean startsWith(List<Rdn> rdns);

    @Positive
    public boolean endsWith(Name n);

    @Positive
    public boolean endsWith(List<Rdn> rdns);

    @Positive
    public Name addAll(Name suffix) throws InvalidNameException;

    @Positive
    public Name addAll(List<Rdn> suffixRdns);

    @Positive
    public Name addAll(int posn, Name suffix) throws InvalidNameException;

    @Positive
    public Name addAll(int posn, List<Rdn> suffixRdns);

    @Positive
    public Name add(String comp) throws InvalidNameException;

    @Positive
    public Name add(Rdn comp);

    @Positive
    public Name add(int posn, String comp) throws InvalidNameException;

    @Positive
    public Name add(int posn, Rdn comp);

    @Positive
    public Object remove(int posn) throws InvalidNameException;

    @Positive
    public List<Rdn> getRdns();

    @Positive
    public Object clone();

    @Positive
    public String toString();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int compareTo(Object obj);

    @Positive
    public int hashCode();
    @Positive
}

// CFWR semantic augmentation - variant 0
