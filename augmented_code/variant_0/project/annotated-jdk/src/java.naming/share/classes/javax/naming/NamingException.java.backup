/*
 * Copyright (c) 1999, 2019, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.nullness.qual.Nullable;

public class NamingException extends Exception {

    protected Name resolvedName;

    @SuppressWarnings("serial")
    protected Object resolvedObj;

    protected Name remainingName;

    protected Throwable rootException;

    public NamingException(String explanation) {
    }

    public NamingException() {
    }

    public Name getResolvedName();

    public Name getRemainingName();

    public Object getResolvedObj();

    public String getExplanation();

    public void setResolvedName(Name name);

    public void setRemainingName(Name name);

    public void setResolvedObj(Object obj);

    public void appendRemainingComponent(String name);

    public void appendRemainingName(Name name);

    public Throwable getRootCause();

    public void setRootCause(Throwable e);

    @Nullable
    public Throwable getCause();

    public Throwable initCause(Throwable cause);

    public String toString();

    public String toString(boolean detail);
}
