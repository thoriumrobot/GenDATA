/*
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
package javax.swing;

import org.checkerframework.checker.interning.qual.Interned;
import org.checkerframework.framework.qual.AnnotatedFor;
import org.checkerframework.checker.nullness.qual.Nullable;
import java.awt.AWTEvent;
import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.Container;
import java.awt.Dialog;
import java.awt.EventQueue;
import java.awt.Frame;
import java.awt.GraphicsEnvironment;
import java.awt.HeadlessException;
import java.awt.Toolkit;
import java.awt.Window;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.HierarchyEvent;
import java.awt.event.HierarchyListener;
import java.awt.event.InputEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.beans.BeanProperty;
import java.beans.JavaBean;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.io.File;
import java.io.IOException;
import java.io.InvalidObjectException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serial;
import java.io.Serializable;
import java.lang.ref.WeakReference;
import java.util.Vector;
import javax.accessibility.Accessible;
import javax.accessibility.AccessibleContext;
import javax.accessibility.AccessibleRole;
import javax.swing.event.EventListenerList;
import javax.swing.filechooser.FileFilter;
import javax.swing.filechooser.FileSystemView;
import javax.swing.filechooser.FileView;
import javax.swing.plaf.FileChooserUI;

@AnnotatedFor({ "interning", "nullness" })
@JavaBean(defaultProperty = "UI", description = "A component which allows for the interactive selection of a file.")
@SwingContainer(false)
@SuppressWarnings("serial")
public class JFileChooser extends JComponent implements Accessible {

    public static final int OPEN_DIALOG;

    public static final int SAVE_DIALOG;

    public static final int CUSTOM_DIALOG;

    public static final int CANCEL_OPTION;

    public static final int APPROVE_OPTION;

    public static final int ERROR_OPTION;

    public static final int FILES_ONLY;

    public static final int DIRECTORIES_ONLY;

    public static final int FILES_AND_DIRECTORIES;

    @Interned
    public static final String CANCEL_SELECTION;

    @Interned
    public static final String APPROVE_SELECTION;

    @Interned
    public static final String APPROVE_BUTTON_TEXT_CHANGED_PROPERTY;

    @Interned
    public static final String APPROVE_BUTTON_TOOL_TIP_TEXT_CHANGED_PROPERTY;

    @Interned
    public static final String APPROVE_BUTTON_MNEMONIC_CHANGED_PROPERTY;

    @Interned
    public static final String CONTROL_BUTTONS_ARE_SHOWN_CHANGED_PROPERTY;

    @Interned
    public static final String DIRECTORY_CHANGED_PROPERTY;

    @Interned
    public static final String SELECTED_FILE_CHANGED_PROPERTY;

    public static final String SELECTED_FILES_CHANGED_PROPERTY;

    @Interned
    public static final String MULTI_SELECTION_ENABLED_CHANGED_PROPERTY;

    @Interned
    public static final String FILE_SYSTEM_VIEW_CHANGED_PROPERTY;

    @Interned
    public static final String FILE_VIEW_CHANGED_PROPERTY;

    public static final String FILE_HIDING_CHANGED_PROPERTY;

    @Interned
    public static final String FILE_FILTER_CHANGED_PROPERTY;

    @Interned
    public static final String FILE_SELECTION_MODE_CHANGED_PROPERTY;

    @Interned
    public static final String ACCESSORY_CHANGED_PROPERTY;

    @Interned
    public static final String ACCEPT_ALL_FILE_FILTER_USED_CHANGED_PROPERTY;

    @Interned
    public static final String DIALOG_TITLE_CHANGED_PROPERTY;

    @Interned
    public static final String DIALOG_TYPE_CHANGED_PROPERTY;

    @Interned
    public static final String CHOOSABLE_FILE_FILTER_CHANGED_PROPERTY;

    public JFileChooser() {
    }

    public JFileChooser(@Nullable String currentDirectoryPath) {
    }

    public JFileChooser(@Nullable File currentDirectory) {
    }

    public JFileChooser(@Nullable FileSystemView fsv) {
    }

    public JFileChooser(@Nullable File currentDirectory, @Nullable FileSystemView fsv) {
    }

    public JFileChooser(@Nullable String currentDirectoryPath, @Nullable FileSystemView fsv) {
    }

    protected void setup(@Nullable FileSystemView view);

    @BeanProperty(bound = false, description = "determines whether automatic drag handling is enabled")
    public void setDragEnabled(boolean b);

    public boolean getDragEnabled();

    @Nullable
    public File getSelectedFile();

    @BeanProperty(preferred = true)
    public void setSelectedFile(@Nullable File file);

    public File[] getSelectedFiles();

    @BeanProperty(description = "The list of selected files if the chooser is in multiple selection mode.")
    public void setSelectedFiles(File @Nullable [] selectedFiles);

    @Nullable
    public File getCurrentDirectory();

    @BeanProperty(preferred = true, description = "The directory that the JFileChooser is showing files of.")
    public void setCurrentDirectory(@Nullable File dir);

    public void changeToParentDirectory();

    public void rescanCurrentDirectory();

    public void ensureFileIsVisible(File f);

    public int showOpenDialog(@Nullable Component parent) throws HeadlessException;

    public int showSaveDialog(@Nullable Component parent) throws HeadlessException;

    @SuppressWarnings("deprecation")
    public int showDialog(@Nullable Component parent, @Nullable String approveButtonText) throws HeadlessException;

    protected JDialog createDialog(@Nullable Component parent) throws HeadlessException;

    public boolean getControlButtonsAreShown();

    @BeanProperty(preferred = true, description = "Sets whether the approve & cancel buttons are shown.")
    public void setControlButtonsAreShown(boolean b);

    public int getDialogType();

    @BeanProperty(preferred = true, enumerationValues = { "JFileChooser.OPEN_DIALOG", "JFileChooser.SAVE_DIALOG", "JFileChooser.CUSTOM_DIALOG" }, description = "The type (open, save, custom) of the JFileChooser.")
    public void setDialogType(int dialogType);

    @BeanProperty(preferred = true, description = "The title of the JFileChooser dialog window.")
    public void setDialogTitle(@Nullable String dialogTitle);

    @Nullable
    public String getDialogTitle();

    @BeanProperty(preferred = true, description = "The tooltip text for the ApproveButton.")
    public void setApproveButtonToolTipText(@Nullable String toolTipText);

    @Nullable
    public String getApproveButtonToolTipText();

    public int getApproveButtonMnemonic();

    @BeanProperty(preferred = true, description = "The mnemonic key accelerator for the ApproveButton.")
    public void setApproveButtonMnemonic(int mnemonic);

    public void setApproveButtonMnemonic(char mnemonic);

    @BeanProperty(preferred = true, description = "The text that goes in the ApproveButton.")
    public void setApproveButtonText(@Nullable String approveButtonText);

    @Nullable
    public String getApproveButtonText();

    @BeanProperty(bound = false)
    public FileFilter[] getChoosableFileFilters();

    @BeanProperty(preferred = true, description = "Adds a filter to the list of user choosable file filters.")
    public void addChoosableFileFilter(@Nullable FileFilter filter);

    public boolean removeChoosableFileFilter(@Nullable FileFilter f);

    public void resetChoosableFileFilters();

    @BeanProperty(bound = false)
    @Nullable
    public FileFilter getAcceptAllFileFilter();

    public boolean isAcceptAllFileFilterUsed();

    @BeanProperty(preferred = true, description = "Sets whether the AcceptAll FileFilter is used as an available choice in the choosable filter list.")
    public void setAcceptAllFileFilterUsed(boolean b);

    @Nullable
    public JComponent getAccessory();

    @BeanProperty(preferred = true, description = "Sets the accessory component on the JFileChooser.")
    public void setAccessory(@Nullable JComponent newAccessory);

    @BeanProperty(preferred = true, enumerationValues = { "JFileChooser.FILES_ONLY", "JFileChooser.DIRECTORIES_ONLY", "JFileChooser.FILES_AND_DIRECTORIES" }, description = "Sets the types of files that the JFileChooser can choose.")
    public void setFileSelectionMode(int mode);

    public int getFileSelectionMode();

    @BeanProperty(bound = false)
    public boolean isFileSelectionEnabled();

    @BeanProperty(bound = false)
    public boolean isDirectorySelectionEnabled();

    @BeanProperty(description = "Sets multiple file selection mode.")
    public void setMultiSelectionEnabled(boolean b);

    public boolean isMultiSelectionEnabled();

    public boolean isFileHidingEnabled();

    @BeanProperty(preferred = true, description = "Sets file hiding on or off.")
    public void setFileHidingEnabled(boolean b);

    @BeanProperty(preferred = true, description = "Sets the File Filter used to filter out files of type.")
    public void setFileFilter(@Nullable FileFilter filter);

    @Nullable
    public FileFilter getFileFilter();

    @BeanProperty(preferred = true, description = "Sets the File View used to get file type information.")
    public void setFileView(@Nullable FileView fileView);

    @Nullable
    public FileView getFileView();

    @Nullable
    public String getName(@Nullable File f);

    @Nullable
    public String getDescription(@Nullable File f);

    @Nullable
    public String getTypeDescription(@Nullable File f);

    @Nullable
    public Icon getIcon(@Nullable File f);

    public boolean isTraversable(@Nullable File f);

    public boolean accept(@Nullable File f);

    @BeanProperty(expert = true, description = "Sets the FileSytemView used to get filesystem information.")
    public void setFileSystemView(@Nullable FileSystemView fsv);

    @Nullable
    public FileSystemView getFileSystemView();

    public void approveSelection();

    public void cancelSelection();

    public void addActionListener(ActionListener l);

    public void removeActionListener(ActionListener l);

    @BeanProperty(bound = false)
    public ActionListener[] getActionListeners();

    @SuppressWarnings("deprecation")
    protected void fireActionPerformed(String command);

    private static class WeakPCL implements PropertyChangeListener {

        public WeakPCL(JFileChooser jfc) {
        }

        public void propertyChange(PropertyChangeEvent ev);
    }

    public void updateUI();

    @BeanProperty(bound = false, expert = true, description = "A string that specifies the name of the L&F class.")
    public String getUIClassID();

    @BeanProperty(bound = false)
    public FileChooserUI getUI();

    protected String paramString();

    protected AccessibleContext accessibleContext;

    @BeanProperty(bound = false)
    public AccessibleContext getAccessibleContext();

    @SuppressWarnings("serial")
    protected class AccessibleJFileChooser extends AccessibleJComponent {

        protected AccessibleJFileChooser() {
        }

        public AccessibleRole getAccessibleRole();
    }

    private class FCHierarchyListener implements HierarchyListener, Serializable {

        @Override
        public void hierarchyChanged(HierarchyEvent e);
    }
}
