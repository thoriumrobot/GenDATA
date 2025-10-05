/*
 * Copyright (c) 1998, 2018, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.jdi;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import com.sun.jdi.AbsentInformationException;
import com.sun.jdi.Location;
import com.sun.jdi.Method;
import com.sun.jdi.ReferenceType;
import com.sun.jdi.VirtualMachine;

public class LocationImpl extends MirrorImpl implements Location {

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int hashCode();

    public int compareTo(Location other);

    public ReferenceType declaringType();

    public Method method();

    public long codeIndex();

    LineInfo getBaseLineInfo(SDE.Stratum stratum);

    LineInfo getLineInfo(SDE.Stratum stratum);

    void addStratumLineInfo(LineInfo lineInfo);

    void addBaseLineInfo(LineInfo lineInfo);

    public String sourceName() throws AbsentInformationException;

    public String sourceName(String stratumID) throws AbsentInformationException;

    String sourceName(SDE.Stratum stratum) throws AbsentInformationException;

    public String sourcePath() throws AbsentInformationException;

    public String sourcePath(String stratumID) throws AbsentInformationException;

    String sourcePath(SDE.Stratum stratum) throws AbsentInformationException;

    public int lineNumber();

    public int lineNumber(String stratumID);

    int lineNumber(SDE.Stratum stratum);

    public String toString();
}
