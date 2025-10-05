/*
 * Copyright (c) 2000, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.beans;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import com.sun.beans.finder.PersistenceDelegateFinder;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.Map;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class Encoder {

    public Encoder() {
    }

    protected void writeObject(Object o);

    public void setExceptionListener(ExceptionListener exceptionListener);

    public ExceptionListener getExceptionListener();

    Object getValue(Expression exp);

    public PersistenceDelegate getPersistenceDelegate(Class<?> type);

    public void setPersistenceDelegate(Class<?> type, PersistenceDelegate delegate);

    public Object remove(Object oldInstance);

    public Object get(Object oldInstance);

    public void writeStatement(Statement oldStm);

    public void writeExpression(Expression oldExp);

    void clear();

    void setAttribute(Object key, Object value);

    Object getAttribute(Object key);
}
