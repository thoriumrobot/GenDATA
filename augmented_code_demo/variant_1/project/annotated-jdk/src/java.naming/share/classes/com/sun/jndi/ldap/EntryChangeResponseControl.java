/*
    @Positive
 * Copyright (c) 1999, 2002, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.jndi.ldap;

    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import java.io.IOException;
    @Positive
import javax.naming.*;
    @Positive
import javax.naming.directory.*;

    @Positive
public final class EntryChangeResponseControl extends BasicControl {

    @Positive
    @Interned
    @Positive
    public static final String OID;

    @Positive
    public static final int ADD;

    @Positive
    public static final int DELETE;

    @Positive
    public static final int MODIFY;

    @Positive
    public static final int RENAME;

    @Positive
    public EntryChangeResponseControl(String id, boolean criticality, byte[] value) throws IOException {
    @Positive
    }

    @Positive
    public int getChangeType();

    @Positive
    public String getPreviousDN();

    @Positive
    public long getChangeNumber();
    @Positive
}

// CFWR semantic augmentation - variant 1
