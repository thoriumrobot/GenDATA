/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1999, 2019, Oracle and/or its affiliates. All rights reserved.
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
package javax.naming;

    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;

    @Positive
public class NamingException extends Exception {

    @Positive
    protected Name resolvedName;

    @Positive
    @SuppressWarnings("serial")
    @Positive
    protected Object resolvedObj;

    @Positive
    protected Name remainingName;

    @Positive
    protected Throwable rootException;

    @Positive
    public NamingException(String explanation) {
    @Positive
    }

    @Positive
    public NamingException() {
    @Positive
    }

    @Positive
    public Name getResolvedName();

    @Positive
    public Name getRemainingName();

    @Positive
    public Object getResolvedObj();

    @Positive
    public String getExplanation();

    @Positive
    public void setResolvedName(Name name);

    @Positive
    public void setRemainingName(Name name);

    @Positive
    public void setResolvedObj(Object obj);

    @Positive
    public void appendRemainingComponent(String name);

    @Positive
    public void appendRemainingName(Name name);

    @Positive
    public Throwable getRootCause();

    @Positive
    public void setRootCause(Throwable e);

    @Positive
    @Nullable
    @Positive
    public Throwable getCause();

    @Positive
    public Throwable initCause(Throwable cause);

    @Positive
    public String toString();

    @Positive
    public String toString(boolean detail);
    @Positive
}
