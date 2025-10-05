/*
 * Copyright (c) 1998, 2021, Oracle and/or its affiliates. All rights reserved.
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
package javax.security.auth;

import org.checkerframework.checker.nonempty.qual.EnsuresNonEmpty;
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
import org.checkerframework.checker.nonempty.qual.NonEmpty;
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.signedness.qual.UnknownSignedness;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import java.util.*;
import java.io.*;
import java.lang.reflect.*;
import java.text.MessageFormat;
import java.security.AccessController;
import java.security.AccessControlContext;
import java.security.DomainCombiner;
import java.security.Permission;
import java.security.PermissionCollection;
import java.security.Principal;
import java.security.PrivilegedAction;
import java.security.PrivilegedExceptionAction;
import java.security.PrivilegedActionException;
import java.security.ProtectionDomain;
import sun.security.util.ResourcesMgr;

public final class Subject implements java.io.Serializable {

    public Subject() {
    }

    public Subject(boolean readOnly, Set<? extends Principal> principals, Set<?> pubCredentials, Set<?> privCredentials) {
    }

    public void setReadOnly();

    public boolean isReadOnly();

    @SuppressWarnings("removal")
    @Deprecated()
    public static Subject getSubject(final AccessControlContext acc);

    @SuppressWarnings("removal")
    public static <T> T doAs(final Subject subject, final java.security.PrivilegedAction<T> action);

    @SuppressWarnings("removal")
    public static <T> T doAs(final Subject subject, final java.security.PrivilegedExceptionAction<T> action) throws java.security.PrivilegedActionException;

    @SuppressWarnings("removal")
    @Deprecated()
    public static <T> T doAsPrivileged(final Subject subject, final java.security.PrivilegedAction<T> action, final java.security.AccessControlContext acc);

    @SuppressWarnings("removal")
    @Deprecated()
    public static <T> T doAsPrivileged(final Subject subject, final java.security.PrivilegedExceptionAction<T> action, final java.security.AccessControlContext acc) throws java.security.PrivilegedActionException;

    public Set<Principal> getPrincipals();

    public <T extends Principal> Set<T> getPrincipals(Class<T> c);

    public Set<Object> getPublicCredentials();

    public Set<Object> getPrivateCredentials();

    public <T> Set<T> getPublicCredentials(Class<T> c);

    public <T> Set<T> getPrivateCredentials(Class<T> c);

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    @Override
    public String toString();

    String toString(boolean includePrivateCredentials);

    @Override
    public int hashCode();

    private static class SecureSet<E> implements Set<E>, java.io.Serializable {

        public int size();

        public Iterator<E> iterator();

        @EnsuresNonEmpty("this")
        public boolean add(E o);

        @SuppressWarnings("removal")
        public boolean remove(@UnknownSignedness Object o);

        @SuppressWarnings("removal")
        @Pure
        @EnsuresNonEmptyIf(result = true, expression = "this")
        public boolean contains(@UnknownSignedness Object o);

        public boolean addAll(Collection<? extends E> c);

        @SuppressWarnings("removal")
        public boolean removeAll(Collection<? extends @UnknownSignedness Object> c);

        @Pure
        public boolean containsAll(Collection<? extends @UnknownSignedness Object> c);

        @SuppressWarnings("removal")
        public boolean retainAll(Collection<? extends @UnknownSignedness Object> c);

        @SuppressWarnings("removal")
        public void clear();

        @EnsuresNonEmptyIf(result = false, expression = "this")
        public boolean isEmpty();

        public Object[] toArray();

        public <T> T[] toArray(T[] a);

        public boolean equals(Object o);

        public int hashCode();
    }

    private class ClassSet<T> extends AbstractSet<T> {

        @Override
        public int size();

        @Override
        public Iterator<T> iterator();

        @Override
        @EnsuresNonEmpty("this")
        public boolean add(T o);
    }

    static final class AuthPermissionHolder {
    }
}
