/*
 * Copyright (c) 1998, 2017, Oracle and/or its affiliates. All rights reserved.
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
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import com.sun.jdi.Field;
import com.sun.jdi.Location;
import com.sun.jdi.NativeMethodException;
import com.sun.jdi.ObjectReference;
import com.sun.jdi.ReferenceType;
import com.sun.jdi.ThreadReference;
import com.sun.jdi.VirtualMachine;
import com.sun.jdi.request.AccessWatchpointRequest;
import com.sun.jdi.request.BreakpointRequest;
import com.sun.jdi.request.ClassPrepareRequest;
import com.sun.jdi.request.ClassUnloadRequest;
import com.sun.jdi.request.DuplicateRequestException;
import com.sun.jdi.request.EventRequest;
import com.sun.jdi.request.EventRequestManager;
import com.sun.jdi.request.ExceptionRequest;
import com.sun.jdi.request.InvalidRequestStateException;
import com.sun.jdi.request.MethodEntryRequest;
import com.sun.jdi.request.MethodExitRequest;
import com.sun.jdi.request.ModificationWatchpointRequest;
import com.sun.jdi.request.MonitorContendedEnterRequest;
import com.sun.jdi.request.MonitorContendedEnteredRequest;
import com.sun.jdi.request.MonitorWaitRequest;
import com.sun.jdi.request.MonitorWaitedRequest;
import com.sun.jdi.request.StepRequest;
import com.sun.jdi.request.ThreadDeathRequest;
import com.sun.jdi.request.ThreadStartRequest;
import com.sun.jdi.request.VMDeathRequest;
import com.sun.jdi.request.WatchpointRequest;

@SuppressWarnings({ "unchecked", "rawtypes" })
class EventRequestManagerImpl extends MirrorImpl implements EventRequestManager {

    static int JDWPtoJDISuspendPolicy(byte jdwpPolicy);

    static byte JDItoJDWPSuspendPolicy(int jdiPolicy);

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int hashCode();

    private abstract class EventRequestImpl extends MirrorImpl implements EventRequest {

        public boolean equals(Object obj);

        public int hashCode();

        abstract int eventCmd();

        InvalidRequestStateException invalidState();

        String state();

        List requestList();

        void delete();

        public boolean isEnabled();

        public void enable();

        public void disable();

        public synchronized void setEnabled(boolean val);

        public synchronized void addCountFilter(int count);

        public void setSuspendPolicy(int policy);

        public int suspendPolicy();

        synchronized void set();

        synchronized void clear();

        public final Object getProperty(Object key);

        public final void putProperty(Object key, Object value);
    }

    abstract class ThreadVisibleEventRequestImpl extends EventRequestImpl {

        public synchronized void addThreadFilter(ThreadReference thread);
    }

    abstract class ClassVisibleEventRequestImpl extends ThreadVisibleEventRequestImpl {

        public synchronized void addClassFilter(ReferenceType clazz);

        public synchronized void addClassFilter(String classPattern);

        public synchronized void addClassExclusionFilter(String classPattern);

        public synchronized void addInstanceFilter(ObjectReference instance);
    }

    class BreakpointRequestImpl extends ClassVisibleEventRequestImpl implements BreakpointRequest {

        public Location location();

        int eventCmd();

        public String toString();
    }

    class ClassPrepareRequestImpl extends ClassVisibleEventRequestImpl implements ClassPrepareRequest {

        int eventCmd();

        public synchronized void addSourceNameFilter(String sourceNamePattern);

        public String toString();
    }

    class ClassUnloadRequestImpl extends ClassVisibleEventRequestImpl implements ClassUnloadRequest {

        int eventCmd();

        public String toString();
    }

    class ExceptionRequestImpl extends ClassVisibleEventRequestImpl implements ExceptionRequest {

        public ReferenceType exception();

        public boolean notifyCaught();

        public boolean notifyUncaught();

        int eventCmd();

        public String toString();
    }

    class MethodEntryRequestImpl extends ClassVisibleEventRequestImpl implements MethodEntryRequest {

        int eventCmd();

        public String toString();
    }

    class MethodExitRequestImpl extends ClassVisibleEventRequestImpl implements MethodExitRequest {

        int eventCmd();

        public String toString();
    }

    class MonitorContendedEnterRequestImpl extends ClassVisibleEventRequestImpl implements MonitorContendedEnterRequest {

        int eventCmd();

        public String toString();
    }

    class MonitorContendedEnteredRequestImpl extends ClassVisibleEventRequestImpl implements MonitorContendedEnteredRequest {

        int eventCmd();

        public String toString();
    }

    class MonitorWaitRequestImpl extends ClassVisibleEventRequestImpl implements MonitorWaitRequest {

        int eventCmd();

        public String toString();
    }

    class MonitorWaitedRequestImpl extends ClassVisibleEventRequestImpl implements MonitorWaitedRequest {

        int eventCmd();

        public String toString();
    }

    class StepRequestImpl extends ClassVisibleEventRequestImpl implements StepRequest {

        public int depth();

        public int size();

        public ThreadReference thread();

        int eventCmd();

        public String toString();
    }

    class ThreadDeathRequestImpl extends ThreadVisibleEventRequestImpl implements ThreadDeathRequest {

        int eventCmd();

        public String toString();
    }

    class ThreadStartRequestImpl extends ThreadVisibleEventRequestImpl implements ThreadStartRequest {

        int eventCmd();

        public String toString();
    }

    abstract class WatchpointRequestImpl extends ClassVisibleEventRequestImpl implements WatchpointRequest {

        public Field field();
    }

    class AccessWatchpointRequestImpl extends WatchpointRequestImpl implements AccessWatchpointRequest {

        int eventCmd();

        public String toString();
    }

    class ModificationWatchpointRequestImpl extends WatchpointRequestImpl implements ModificationWatchpointRequest {

        int eventCmd();

        public String toString();
    }

    class VMDeathRequestImpl extends EventRequestImpl implements VMDeathRequest {

        int eventCmd();

        public String toString();
    }

    public ClassPrepareRequest createClassPrepareRequest();

    public ClassUnloadRequest createClassUnloadRequest();

    public ExceptionRequest createExceptionRequest(ReferenceType refType, boolean notifyCaught, boolean notifyUncaught);

    public StepRequest createStepRequest(ThreadReference thread, int size, int depth);

    public ThreadDeathRequest createThreadDeathRequest();

    public ThreadStartRequest createThreadStartRequest();

    public MethodEntryRequest createMethodEntryRequest();

    public MethodExitRequest createMethodExitRequest();

    public MonitorContendedEnterRequest createMonitorContendedEnterRequest();

    public MonitorContendedEnteredRequest createMonitorContendedEnteredRequest();

    public MonitorWaitRequest createMonitorWaitRequest();

    public MonitorWaitedRequest createMonitorWaitedRequest();

    public BreakpointRequest createBreakpointRequest(Location location);

    public AccessWatchpointRequest createAccessWatchpointRequest(Field field);

    public ModificationWatchpointRequest createModificationWatchpointRequest(Field field);

    public VMDeathRequest createVMDeathRequest();

    public void deleteEventRequest(EventRequest eventRequest);

    public void deleteEventRequests(List<? extends EventRequest> eventRequests);

    public void deleteAllBreakpoints();

    public List<StepRequest> stepRequests();

    public List<ClassPrepareRequest> classPrepareRequests();

    public List<ClassUnloadRequest> classUnloadRequests();

    public List<ThreadStartRequest> threadStartRequests();

    public List<ThreadDeathRequest> threadDeathRequests();

    public List<ExceptionRequest> exceptionRequests();

    public List<BreakpointRequest> breakpointRequests();

    public List<AccessWatchpointRequest> accessWatchpointRequests();

    public List<ModificationWatchpointRequest> modificationWatchpointRequests();

    public List<MethodEntryRequest> methodEntryRequests();

    public List<MethodExitRequest> methodExitRequests();

    public List<MonitorContendedEnterRequest> monitorContendedEnterRequests();

    public List<MonitorContendedEnteredRequest> monitorContendedEnteredRequests();

    public List<MonitorWaitRequest> monitorWaitRequests();

    public List<MonitorWaitedRequest> monitorWaitedRequests();

    public List<VMDeathRequest> vmDeathRequests();

    List<? extends EventRequest> unmodifiableRequestList(int eventCmd);

    EventRequest request(int eventCmd, int requestId);
}
