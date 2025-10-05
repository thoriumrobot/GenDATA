/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.security;

    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nonempty.qual.NonEmpty;
    @Positive
import java.io.InvalidObjectException;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.ObjectStreamField;
    @Positive
import java.io.Serializable;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.Hashtable;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import java.util.concurrent.ConcurrentHashMap;

    @Positive
public final class Permissions extends PermissionCollection implements Serializable {

    @Positive
    public Permissions() {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public void add(Permission permission);

    @Positive
    @Override
    @Positive
    public boolean implies(Permission permission);

    @Positive
    @Override
    @Positive
    public Enumeration<Permission> elements();
    @Positive
}

    @Positive
final class PermissionsEnumerator implements Enumeration<Permission> {

    @Positive
    @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
    public boolean hasMoreElements();

    @Positive
    public Permission nextElement(@NonEmpty PermissionsEnumerator this);
    @Positive
}

    @Positive
final class PermissionsHash extends PermissionCollection implements Serializable {

    @Positive
    @Override
    @Positive
    public void add(Permission permission);

    @Positive
    @Override
    @Positive
    public boolean implies(Permission permission);

    @Positive
    @Override
    @Positive
    public Enumeration<Permission> elements();
    @Positive
}
