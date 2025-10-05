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
package javax.swing;

    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import java.awt.AWTEvent;
    @Positive
import java.awt.BorderLayout;
    @Positive
import java.awt.Component;
    @Positive
import java.awt.Container;
    @Positive
import java.awt.Dialog;
    @Positive
import java.awt.EventQueue;
    @Positive
import java.awt.Frame;
    @Positive
import java.awt.GraphicsEnvironment;
    @Positive
import java.awt.HeadlessException;
    @Positive
import java.awt.Toolkit;
    @Positive
import java.awt.Window;
    @Positive
import java.awt.event.ActionEvent;
    @Positive
import java.awt.event.ActionListener;
    @Positive
import java.awt.event.HierarchyEvent;
    @Positive
import java.awt.event.HierarchyListener;
    @Positive
import java.awt.event.InputEvent;
    @Positive
import java.awt.event.WindowAdapter;
    @Positive
import java.awt.event.WindowEvent;
    @Positive
import java.beans.BeanProperty;
    @Positive
import java.beans.JavaBean;
    @Positive
import java.beans.PropertyChangeEvent;
    @Positive
import java.beans.PropertyChangeListener;
    @Positive
import java.io.File;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InvalidObjectException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.Serial;
    @Positive
import java.io.Serializable;
    @Positive
import java.lang.ref.WeakReference;
    @Positive
import java.util.Vector;
    @Positive
import javax.accessibility.Accessible;
    @Positive
import javax.accessibility.AccessibleContext;
    @Positive
import javax.accessibility.AccessibleRole;
    @Positive
import javax.swing.event.EventListenerList;
    @Positive
import javax.swing.filechooser.FileFilter;
    @Positive
import javax.swing.filechooser.FileSystemView;
    @Positive
import javax.swing.filechooser.FileView;
    @Positive
import javax.swing.plaf.FileChooserUI;

    @Positive
@AnnotatedFor({ "interning", "nullness" })
    @Positive
@JavaBean(defaultProperty = "UI", description = "A component which allows for the interactive selection of a file.")
    @Positive
@SwingContainer(false)
    @Positive
@SuppressWarnings("serial")
    @Positive
public class JFileChooser extends JComponent implements Accessible {

    @Positive
    public static final int OPEN_DIALOG;

    @Positive
    public static final int SAVE_DIALOG;

    @Positive
    public static final int CUSTOM_DIALOG;

    @Positive
    public static final int CANCEL_OPTION;

    @Positive
    public static final int APPROVE_OPTION;

    @Positive
    public static final int ERROR_OPTION;

    @Positive
    public static final int FILES_ONLY;

    @Positive
    public static final int DIRECTORIES_ONLY;

    @Positive
    public static final int FILES_AND_DIRECTORIES;

    @Positive
    @Interned
    @Positive
    public static final String CANCEL_SELECTION;

    @Positive
    @Interned
    @Positive
    public static final String APPROVE_SELECTION;

    @Positive
    @Interned
    @Positive
    public static final String APPROVE_BUTTON_TEXT_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String APPROVE_BUTTON_TOOL_TIP_TEXT_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String APPROVE_BUTTON_MNEMONIC_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String CONTROL_BUTTONS_ARE_SHOWN_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String DIRECTORY_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String SELECTED_FILE_CHANGED_PROPERTY;

    @Positive
    public static final String SELECTED_FILES_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String MULTI_SELECTION_ENABLED_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String FILE_SYSTEM_VIEW_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String FILE_VIEW_CHANGED_PROPERTY;

    @Positive
    public static final String FILE_HIDING_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String FILE_FILTER_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String FILE_SELECTION_MODE_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String ACCESSORY_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String ACCEPT_ALL_FILE_FILTER_USED_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String DIALOG_TITLE_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String DIALOG_TYPE_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String CHOOSABLE_FILE_FILTER_CHANGED_PROPERTY;

    @Positive
    public JFileChooser() {
    @Positive
    }

    @Positive
    public JFileChooser(@Nullable String currentDirectoryPath) {
    @Positive
    }

    @Positive
    public JFileChooser(@Nullable File currentDirectory) {
    @Positive
    }

    @Positive
    public JFileChooser(@Nullable FileSystemView fsv) {
    @Positive
    }

    @Positive
    public JFileChooser(@Nullable File currentDirectory, @Nullable FileSystemView fsv) {
    @Positive
    }

    @Positive
    public JFileChooser(@Nullable String currentDirectoryPath, @Nullable FileSystemView fsv) {
    @Positive
    }

    @Positive
    protected void setup(@Nullable FileSystemView view);

    @Positive
    @BeanProperty(bound = false, description = "determines whether automatic drag handling is enabled")
    @Positive
    public void setDragEnabled(boolean b);

    @Positive
    public boolean getDragEnabled();

    @Positive
    @Nullable
    @Positive
    public File getSelectedFile();

    @Positive
    @BeanProperty(preferred = true)
    @Positive
    public void setSelectedFile(@Nullable File file);

    @Positive
    public File[] getSelectedFiles();

    @Positive
    @BeanProperty(description = "The list of selected files if the chooser is in multiple selection mode.")
    @Positive
    public void setSelectedFiles(File @Nullable [] selectedFiles);

    @Positive
    @Nullable
    @Positive
    public File getCurrentDirectory();

    @Positive
    @BeanProperty(preferred = true, description = "The directory that the JFileChooser is showing files of.")
    @Positive
    public void setCurrentDirectory(@Nullable File dir);

    @Positive
    public void changeToParentDirectory();

    @Positive
    public void rescanCurrentDirectory();

    @Positive
    public void ensureFileIsVisible(File f);

    @Positive
    public int showOpenDialog(@Nullable Component parent) throws HeadlessException;

    @Positive
    public int showSaveDialog(@Nullable Component parent) throws HeadlessException;

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    public int showDialog(@Nullable Component parent, @Nullable String approveButtonText) throws HeadlessException;

    @Positive
    protected JDialog createDialog(@Nullable Component parent) throws HeadlessException;

    @Positive
    public boolean getControlButtonsAreShown();

    @Positive
    @BeanProperty(preferred = true, description = "Sets whether the approve & cancel buttons are shown.")
    @Positive
    public void setControlButtonsAreShown(boolean b);

    @Positive
    public int getDialogType();

    @Positive
    @BeanProperty(preferred = true, enumerationValues = { "JFileChooser.OPEN_DIALOG", "JFileChooser.SAVE_DIALOG", "JFileChooser.CUSTOM_DIALOG" }, description = "The type (open, save, custom) of the JFileChooser.")
    @Positive
    public void setDialogType(int dialogType);

    @Positive
    @BeanProperty(preferred = true, description = "The title of the JFileChooser dialog window.")
    @Positive
    public void setDialogTitle(@Nullable String dialogTitle);

    @Positive
    @Nullable
    @Positive
    public String getDialogTitle();

    @Positive
    @BeanProperty(preferred = true, description = "The tooltip text for the ApproveButton.")
    @Positive
    public void setApproveButtonToolTipText(@Nullable String toolTipText);

    @Positive
    @Nullable
    @Positive
    public String getApproveButtonToolTipText();

    @Positive
    public int getApproveButtonMnemonic();

    @Positive
    @BeanProperty(preferred = true, description = "The mnemonic key accelerator for the ApproveButton.")
    @Positive
    public void setApproveButtonMnemonic(int mnemonic);

    @Positive
    public void setApproveButtonMnemonic(char mnemonic);

    @Positive
    @BeanProperty(preferred = true, description = "The text that goes in the ApproveButton.")
    @Positive
    public void setApproveButtonText(@Nullable String approveButtonText);

    @Positive
    @Nullable
    @Positive
    public String getApproveButtonText();

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public FileFilter[] getChoosableFileFilters();

    @Positive
    @BeanProperty(preferred = true, description = "Adds a filter to the list of user choosable file filters.")
    @Positive
    public void addChoosableFileFilter(@Nullable FileFilter filter);

    @Positive
    public boolean removeChoosableFileFilter(@Nullable FileFilter f);

    @Positive
    public void resetChoosableFileFilters();

    @Positive
    @BeanProperty(bound = false)
    @Positive
    @Nullable
    @Positive
    public FileFilter getAcceptAllFileFilter();

    @Positive
    public boolean isAcceptAllFileFilterUsed();

    @Positive
    @BeanProperty(preferred = true, description = "Sets whether the AcceptAll FileFilter is used as an available choice in the choosable filter list.")
    @Positive
    public void setAcceptAllFileFilterUsed(boolean b);

    @Positive
    @Nullable
    @Positive
    public JComponent getAccessory();

    @Positive
    @BeanProperty(preferred = true, description = "Sets the accessory component on the JFileChooser.")
    @Positive
    public void setAccessory(@Nullable JComponent newAccessory);

    @Positive
    @BeanProperty(preferred = true, enumerationValues = { "JFileChooser.FILES_ONLY", "JFileChooser.DIRECTORIES_ONLY", "JFileChooser.FILES_AND_DIRECTORIES" }, description = "Sets the types of files that the JFileChooser can choose.")
    @Positive
    public void setFileSelectionMode(int mode);

    @Positive
    public int getFileSelectionMode();

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public boolean isFileSelectionEnabled();

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public boolean isDirectorySelectionEnabled();

    @Positive
    @BeanProperty(description = "Sets multiple file selection mode.")
    @Positive
    public void setMultiSelectionEnabled(boolean b);

    @Positive
    public boolean isMultiSelectionEnabled();

    @Positive
    public boolean isFileHidingEnabled();

    @Positive
    @BeanProperty(preferred = true, description = "Sets file hiding on or off.")
    @Positive
    public void setFileHidingEnabled(boolean b);

    @Positive
    @BeanProperty(preferred = true, description = "Sets the File Filter used to filter out files of type.")
    @Positive
    public void setFileFilter(@Nullable FileFilter filter);

    @Positive
    @Nullable
    @Positive
    public FileFilter getFileFilter();

    @Positive
    @BeanProperty(preferred = true, description = "Sets the File View used to get file type information.")
    @Positive
    public void setFileView(@Nullable FileView fileView);

    @Positive
    @Nullable
    @Positive
    public FileView getFileView();

    @Positive
    @Nullable
    @Positive
    public String getName(@Nullable File f);

    @Positive
    @Nullable
    @Positive
    public String getDescription(@Nullable File f);

    @Positive
    @Nullable
    @Positive
    public String getTypeDescription(@Nullable File f);

    @Positive
    @Nullable
    @Positive
    public Icon getIcon(@Nullable File f);

    @Positive
    public boolean isTraversable(@Nullable File f);

    @Positive
    public boolean accept(@Nullable File f);

    @Positive
    @BeanProperty(expert = true, description = "Sets the FileSytemView used to get filesystem information.")
    @Positive
    public void setFileSystemView(@Nullable FileSystemView fsv);

    @Positive
    @Nullable
    @Positive
    public FileSystemView getFileSystemView();

    @Positive
    public void approveSelection();

    @Positive
    public void cancelSelection();

    @Positive
    public void addActionListener(ActionListener l);

    @Positive
    public void removeActionListener(ActionListener l);

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public ActionListener[] getActionListeners();

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    protected void fireActionPerformed(String command);

    @Positive
    private static class WeakPCL implements PropertyChangeListener {

    @Positive
        public WeakPCL(JFileChooser jfc) {
    @Positive
        }

    @Positive
        public void propertyChange(PropertyChangeEvent ev);
    @Positive
    }

    @Positive
    public void updateUI();

    @Positive
    @BeanProperty(bound = false, expert = true, description = "A string that specifies the name of the L&F class.")
    @Positive
    public String getUIClassID();

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public FileChooserUI getUI();

    @Positive
    protected String paramString();

    @Positive
    protected AccessibleContext accessibleContext;

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public AccessibleContext getAccessibleContext();

    @Positive
    @SuppressWarnings("serial")
    @Positive
    protected class AccessibleJFileChooser extends AccessibleJComponent {

    @Positive
        protected AccessibleJFileChooser() {
    @Positive
        }

    @Positive
        public AccessibleRole getAccessibleRole();
    @Positive
    }

    @Positive
    private class FCHierarchyListener implements HierarchyListener, Serializable {

    @Positive
        @Override
    @Positive
        public void hierarchyChanged(HierarchyEvent e);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
