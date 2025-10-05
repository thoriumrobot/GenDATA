/*
    @Positive
 * Copyright (c) 1999, 2007, Oracle and/or its affiliates. All rights reserved.
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
package javax.management.monitor;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import javax.management.ObjectName;

    @Positive
public interface MonitorMBean {

    @Positive
    public void start();

    @Positive
    public void stop();

    @Positive
    public void addObservedObject(ObjectName object) throws java.lang.IllegalArgumentException;

    @Positive
    public void removeObservedObject(ObjectName object);

    @Positive
    @Pure
    @Positive
    public boolean containsObservedObject(ObjectName object);

    @Positive
    public ObjectName[] getObservedObjects();

    @Positive
    @Deprecated
    @Positive
    public ObjectName getObservedObject();

    @Positive
    @Deprecated
    @Positive
    public void setObservedObject(ObjectName object);

    @Positive
    public String getObservedAttribute();

    @Positive
    public void setObservedAttribute(String attribute);

    @Positive
    public long getGranularityPeriod();

    @Positive
    public void setGranularityPeriod(long period) throws java.lang.IllegalArgumentException;

    @Positive
    public boolean isActive();
    @Positive
}

// CFWR semantic augmentation - variant 0
